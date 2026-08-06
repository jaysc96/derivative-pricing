"""U8: nothing provider-shaped crosses the adapter boundary, and bad data stops there.

Every test here runs against a recorded fixture. No test in this file reaches
the network — the provider is an unofficial scraper and a suite that depends on
Yahoo being reachable fails for reasons that have nothing to do with the code.
"""

import socket
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from marketdata import (
    ChainSnapshot,
    MalformedResponse,
    NotSupported,
    ProviderUnavailable,
    QuoteRecord,
    RateLimited,
    YFinanceAdapter,
)
from marketdata.yfinance_adapter import classify

# Shaped like a real yfinance frame, including the parts that bite: a zero bid
# on a thin contract, a null volume, and lastTradeDate as a tz-aware stamp.
CALLS = pd.DataFrame(
    [
        {
            "contractSymbol": "SPY260918C00600000",
            "strike": 600.0,
            "bid": 12.5,
            "ask": 12.9,
            "lastPrice": 12.7,
            "volume": 431,
            "openInterest": 2210,
            "impliedVolatility": 0.1842,
            "lastTradeDate": pd.Timestamp("2026-08-05 19:58:11", tz="UTC"),
        },
        {
            "contractSymbol": "SPY260918C00900000",
            "strike": 900.0,
            "bid": 0.0,
            "ask": 0.05,
            "lastPrice": 0.02,
            "volume": None,
            "openInterest": 4,
            "impliedVolatility": 0.9917,
            "lastTradeDate": pd.Timestamp("2026-07-28 14:02:00", tz="UTC"),
        },
    ]
)

PUTS = pd.DataFrame(
    [
        {
            "contractSymbol": "SPY260918P00600000",
            "strike": 600.0,
            "bid": 9.1,
            "ask": 9.4,
            "lastPrice": 9.25,
            "volume": 88,
            "openInterest": 1043,
            "impliedVolatility": 0.2011,
            "lastTradeDate": pd.Timestamp("2026-08-05 19:44:03", tz="UTC"),
        }
    ]
)


class FakeChain:
    def __init__(self, calls, puts):
        self.calls, self.puts = calls, puts


class FakeTicker:
    """A recorded provider. Raises whatever it is told to raise."""

    def __init__(
        self, *, calls=CALLS, puts=PUTS, expiries=("2026-09-18",), error=None, history=None,
        fast_info=None, info=None,
    ):
        self._calls, self._puts, self._expiries = calls, puts, expiries
        self._error, self._history = error, history
        self._fast_info = fast_info if fast_info is not None else {}
        self._info = info if info is not None else {}

    @property
    def options(self):
        if self._error:
            raise self._error
        return self._expiries

    @property
    def fast_info(self):
        if self._error:
            raise self._error
        return self._fast_info

    @property
    def info(self):
        if self._error:
            raise self._error
        return self._info

    def option_chain(self, _expiry):
        if self._error:
            raise self._error
        return FakeChain(self._calls, self._puts)

    def history(self, **_kwargs):
        if self._error:
            raise self._error
        return self._history


def adapter_for(ticker):
    return YFinanceAdapter(ticker_factory=lambda _symbol: ticker)


# --------------------------------------------------------------------------
# The boundary holds
# --------------------------------------------------------------------------


def test_chain_returns_neutral_records_only():
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))

    assert isinstance(snapshot, ChainSnapshot)
    assert all(isinstance(quote, QuoteRecord) for quote in snapshot.quotes)
    assert len(snapshot.quotes) == 3
    assert {q.option_type for q in snapshot.quotes} == {"call", "put"}


def test_no_provider_shaped_value_escapes():
    """No DataFrame, no Yahoo field name, on the snapshot or on any record."""
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))

    yahoo_names = {"contractSymbol", "lastPrice", "openInterest", "impliedVolatility", "lastTradeDate"}
    for quote in snapshot.quotes:
        assert not isinstance(quote, pd.DataFrame)
        assert yahoo_names.isdisjoint(vars(quote))
    assert not isinstance(snapshot.quotes, pd.DataFrame)


def test_provider_iv_is_carried_through():
    """Free from the provider, and the difference against ours is a product."""
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))
    atm = next(q for q in snapshot.quotes if q.contract_symbol.endswith("C00600000"))
    assert atm.provider_iv == pytest.approx(0.1842)


def test_recency_marker_is_distinct_from_capture_time():
    """It times the last trade, not the quote — the two must not be conflated."""
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))
    stale = next(q for q in snapshot.quotes if q.contract_symbol.endswith("C00900000"))

    assert stale.last_trade_at is not None
    assert stale.last_trade_at < snapshot.captured_at
    assert (snapshot.captured_at - stale.last_trade_at).days >= 1


def test_two_sided_is_a_question_about_the_bid():
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))
    quoted = next(q for q in snapshot.quotes if q.contract_symbol.endswith("C00600000"))
    unbid = next(q for q in snapshot.quotes if q.contract_symbol.endswith("C00900000"))

    assert quoted.two_sided and quoted.mid == pytest.approx(12.7)
    assert not unbid.two_sided and unbid.mid is None
    assert unbid.ask == pytest.approx(0.05), "ask is present; only the bid is missing"


# --------------------------------------------------------------------------
# Errors become ours
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised,expected",
    [
        (RuntimeError("429 Too Many Requests"), RateLimited),
        (RuntimeError("Read timed out"), ProviderUnavailable),
        (RuntimeError("Connection refused"), ProviderUnavailable),
        (ValueError("something else entirely"), ProviderUnavailable),
    ],
)
def test_library_exceptions_become_interface_errors(raised, expected):
    adapter = adapter_for(FakeTicker(error=raised))
    with pytest.raises(expected):
        adapter.option_chain("SPY", date(2026, 9, 18))
    with pytest.raises(expected):
        adapter.expiries("SPY")


def test_dated_chain_is_refused_rather_than_faked():
    """A provider without history says so; it does not quietly return today's."""
    with pytest.raises(NotSupported):
        adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18), as_of=date(2026, 1, 5))


@pytest.mark.parametrize(
    "message",
    [
        "Failed to generate request signature",
        "Could not separate calls from puts",
        "Corporate action data missing",
        "accurate pricing unavailable for this contract",
    ],
)
def test_an_unrelated_error_is_not_read_as_a_rate_limit(message):
    """The substring 'rate' lives inside four ordinary words.

    Reading any of them as throttling costs three attempts with exponential
    backoff and then files a `rate_limited` code, which is the signal the
    fallback decision is built on. A wrong one there argues for building a
    second provider because Yahoo is throttling us, when it is not.
    """
    assert not isinstance(classify(RuntimeError(message)), RateLimited)


@pytest.mark.parametrize(
    "message",
    ["429 Too Many Requests", "Rate limit exceeded", "ratelimited", "rate-limit hit"],
)
def test_a_real_rate_limit_is_still_recognised(message):
    assert isinstance(classify(RuntimeError(message)), RateLimited)


def test_a_provider_call_runs_under_a_socket_timeout():
    """An unattended job has no worse failure than a hang.

    A blocked run records nothing at all — neither productive nor
    unproductive — so the fallback trigger never fires and history stops with
    no signal that it has.
    """
    seen = []

    class Observing(FakeTicker):
        @property
        def options(self):
            seen.append(socket.getdefaulttimeout())
            return super().options

    before = socket.getdefaulttimeout()
    YFinanceAdapter(lambda _symbol: Observing(), timeout=12.5).expiries("SPY")

    assert seen == [12.5], "the bound was not in force during the call"
    assert socket.getdefaulttimeout() == before, "the global default was not restored"


def test_our_own_error_is_not_reclassified_into_a_failed_request():
    """A missing dependency carries an instruction; classify() would erase it.

    Everything the adapter raises is already in the interface's vocabulary, so
    passing it back through the translator turns "install the data extra" into
    "provider request failed (ProviderUnavailable)".
    """
    def refuse(_symbol):
        raise ProviderUnavailable("yfinance is not installed; install the 'data' extra")

    with pytest.raises(ProviderUnavailable, match="install the 'data' extra"):
        YFinanceAdapter(refuse).expiries("SPY")


# --------------------------------------------------------------------------
# Malformed data stops at the edge
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column,value",
    [("bid", -1.0), ("ask", -0.01), ("strike", -600.0), ("strike", 0.0), ("lastPrice", -5.0)],
)
def test_negative_or_impossible_prices_are_rejected(column, value):
    bad = CALLS.copy()
    bad.loc[0, column] = value
    with pytest.raises(MalformedResponse):
        adapter_for(FakeTicker(calls=bad)).option_chain("SPY", date(2026, 9, 18))


def test_out_of_range_price_is_rejected():
    bad = CALLS.copy()
    bad.loc[0, "ask"] = 5_000_000.0
    with pytest.raises(MalformedResponse):
        adapter_for(FakeTicker(calls=bad)).option_chain("SPY", date(2026, 9, 18))


def test_missing_strike_is_rejected():
    bad = CALLS.copy()
    bad.loc[0, "strike"] = None
    with pytest.raises(MalformedResponse):
        adapter_for(FakeTicker(calls=bad)).option_chain("SPY", date(2026, 9, 18))


def test_renamed_field_is_caught_rather_than_read_as_a_thin_chain():
    """The scraper's real failure mode.

    A renamed column arrives as absent data, which looks like an illiquid chain
    rather than a broken one. Without this check history stops accruing while
    every run reports success.
    """
    renamed = CALLS.rename(columns={"bid": "bidPrice"})
    with pytest.raises(MalformedResponse, match="missing"):
        adapter_for(FakeTicker(calls=renamed)).option_chain("SPY", date(2026, 9, 18))


def test_absent_values_are_none_rather_than_errors():
    """A missing bid is ordinary — 31% of contracts in the spike."""
    sparse = CALLS.copy()
    sparse.loc[0, "bid"] = None
    sparse.loc[0, "impliedVolatility"] = 0.0

    snapshot = adapter_for(FakeTicker(calls=sparse)).option_chain("SPY", date(2026, 9, 18))
    quote = next(q for q in snapshot.quotes if q.contract_symbol.endswith("C00600000"))

    assert quote.bid is None
    assert quote.provider_iv is None, "a provider zero means 'could not compute', not zero vol"
    assert quote.volume is None or isinstance(quote.volume, int)


# --------------------------------------------------------------------------
# Expiries and underlying history
# --------------------------------------------------------------------------


def test_expiries_are_dates_not_strings():
    got = adapter_for(FakeTicker(expiries=("2026-09-18", "2026-10-16"))).expiries("SPY")
    assert got == (date(2026, 9, 18), date(2026, 10, 16))


def test_malformed_expiry_list_is_rejected():
    with pytest.raises(MalformedResponse):
        adapter_for(FakeTicker(expiries=("not-a-date",))).expiries("SPY")


def test_underlying_history_returns_neutral_bars():
    frame = pd.DataFrame(
        [{"Open": 600.0, "High": 606.0, "Low": 598.0, "Close": 604.0, "Volume": 71_000_000}],
        index=[pd.Timestamp("2026-08-04")],
    )
    bars = adapter_for(FakeTicker(history=frame)).underlying_history(
        "SPY", date(2026, 8, 1), date(2026, 8, 5)
    )
    assert len(bars) == 1
    assert bars[0].bar_date == date(2026, 8, 4)
    assert bars[0].close == pytest.approx(604.0)


def test_snapshot_records_origin_and_as_of():
    snapshot = adapter_for(FakeTicker()).option_chain("SPY", date(2026, 9, 18))
    assert snapshot.origin == "live"
    assert snapshot.as_of == snapshot.captured_at.date()
    assert snapshot.captured_at.tzinfo is timezone.utc


# --------------------------------------------------------------------------
# Risk-free rate and dividend yield
# --------------------------------------------------------------------------


def test_risk_free_rate_scales_the_treasury_quote_to_a_decimal():
    """Yahoo quotes ^IRX as a bare number (3.73 meaning 3.73%), not a fraction."""
    rate = adapter_for(FakeTicker(fast_info={"lastPrice": 3.73})).risk_free_rate()
    assert rate == pytest.approx(0.0373)


def test_dividend_yield_reads_the_trailing_annual_field():
    """Not `dividendYield` — measured against the tracked set, that field is
    missing for a near-zero payer while `trailingAnnualDividendYield` is
    present for every one of them, and already a decimal fraction."""
    yield_ = adapter_for(FakeTicker(info={"trailingAnnualDividendYield": 0.0074})).dividend_yield("SPY")
    assert yield_ == pytest.approx(0.0074)


def test_dividend_yield_absent_is_none_not_zero():
    """A missing field and a confirmed non-payer are not distinguished (see the
    adapter's own docstring) — but both must come back as None, not a
    fabricated 0.0 that looks like a measured fact."""
    assert adapter_for(FakeTicker(info={})).dividend_yield("NVDA") is None


def test_a_missing_rate_reading_is_none():
    assert adapter_for(FakeTicker(fast_info={})).risk_free_rate() is None


@pytest.mark.parametrize(
    "raised,expected",
    [
        (RuntimeError("429 Too Many Requests"), RateLimited),
        (RuntimeError("Read timed out"), ProviderUnavailable),
    ],
)
def test_rate_and_yield_fetch_failures_become_interface_errors(raised, expected):
    adapter = adapter_for(FakeTicker(error=raised))
    with pytest.raises(expected):
        adapter.risk_free_rate()
    with pytest.raises(expected):
        adapter.dividend_yield("SPY")


def test_a_dated_rate_or_yield_is_refused_rather_than_faked():
    with pytest.raises(NotSupported):
        adapter_for(FakeTicker()).risk_free_rate(as_of=date(2026, 1, 5))
    with pytest.raises(NotSupported):
        adapter_for(FakeTicker()).dividend_yield("SPY", as_of=date(2026, 1, 5))


def test_an_implausible_rate_is_rejected_as_a_likely_units_error():
    """A units error (a percentage handed through as a whole number) lands far
    outside any real rate; the validator catches the shape, not the sign."""
    with pytest.raises(MalformedResponse):
        adapter_for(FakeTicker(fast_info={"lastPrice": 250.0})).risk_free_rate()


def test_a_negative_rate_is_accepted():
    """Short rates have genuinely gone negative for years at a time; rejecting
    on sign alone would be a wrong assumption baked into the validator."""
    rate = adapter_for(FakeTicker(fast_info={"lastPrice": -0.5})).risk_free_rate()
    assert rate == pytest.approx(-0.005)
