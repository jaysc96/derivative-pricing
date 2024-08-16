import numpy as np
import scipy.stats
from datetime import datetime as dt
N = scipy.stats.norm.cdf

class Stock_Option:
    def __init__(self, type, initial_price, strike_price, interest_rate, volatility, dividend_rate, time, kind, start_date, end_date) -> None:
        self.type = type
        self.S0 = initial_price
        self.X = strike_price
        self.r = interest_rate
        self.sig = volatility
        self.y = dividend_rate
        self.T = time
        self.kind = kind
        self.start_date = start_date
        self.end_date = end_date

    def BSM(self):
        phi = 1
        if self.kind == 'Put':
            phi = -1

        zp = (np.log(self.S0 / self.X) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) + self.sig * np.sqrt(self.T) / 2
        zm = (np.log(self.S0 / self.X) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) - self.sig * np.sqrt(self.T) / 2

        O = phi * (self.S0 * np.exp(-self.y * self.T) * N(phi * zp) - self.X * np.exp(-self.r * self.T) * N(phi * zm))
        return O

    def priceOption(self, method='BSM'):
        if method=='BSM':
            return self.BSM()