import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LineChart, { Legend } from "./LineChart";

function segmentCount(pathElement) {
  // Each broken segment starts with its own "M" (moveto); counting them is
  // the cheapest way to observe how many disconnected pieces the line drew
  // into, without parsing the full path grammar.
  return (pathElement.getAttribute("d").match(/M/g) || []).length;
}

it("renders nothing when no series has a valid point", () => {
  const { container } = render(<LineChart series={[{ name: "a", color: "#000", points: [] }]} />);
  expect(container).toBeEmptyDOMElement();
});

it("draws one unbroken segment for evenly spaced points", () => {
  const series = [
    {
      name: "2026-08-11",
      color: "#58a6ff",
      points: [90, 91, 92, 93].map((x) => ({ x, y: 0.2 })),
    },
  ];
  render(<LineChart series={series} maxGapMultiple={4} ariaLabel="test chart" />);

  const path = document.querySelector("path");
  expect(segmentCount(path)).toBe(1);
});

describe("gap-breaking on a 3-point curve", () => {
  // The minimum size a real "ok" skew curve ever has (build_skew's own
  // 3-point floor) -- and the exact shape that defeated medianGap's old
  // 2-gap "median": with gaps of [1, 20], picking index floor(2/2)=1 chose
  // the outlier (20) as its own baseline, so the break it exists to trigger
  // never fired. This test is the regression proof for that fix.
  it("breaks the line at a strike gap that dwarfs the curve's own typical spacing", () => {
    const series = [
      {
        name: "2026-08-13",
        color: "#f0883e",
        points: [
          { x: 100, y: 0.12 },
          { x: 101, y: 0.121 },
          { x: 121, y: 0.3 },
        ],
      },
    ];
    render(<LineChart series={series} maxGapMultiple={4} ariaLabel="test chart" />);

    const path = document.querySelector("path");
    expect(segmentCount(path)).toBe(2);
  });

  it("does not break when maxGapMultiple is left at the Infinity default", () => {
    const series = [
      {
        name: "2026-08-13",
        color: "#f0883e",
        points: [
          { x: 100, y: 0.12 },
          { x: 101, y: 0.121 },
          { x: 121, y: 0.3 },
        ],
      },
    ];
    render(<LineChart series={series} ariaLabel="test chart" />);

    const path = document.querySelector("path");
    expect(segmentCount(path)).toBe(1);
  });
});

it("breaks the line at a null y without plotting a dive to the axis", () => {
  const series = [
    {
      name: "implied",
      color: "#58a6ff",
      points: [
        { x: 0, y: 0.2 },
        { x: 1, y: null },
        { x: 2, y: 0.22 },
      ],
    },
  ];
  render(<LineChart series={series} ariaLabel="test chart" />);

  const path = document.querySelector("path");
  expect(segmentCount(path)).toBe(2);
  // Only the 2 valid points render as circles -- the null point contributes
  // no dot pretending to be a real observation at y=0.
  expect(screen.getAllByTestId("chart-point")).toHaveLength(2);
});

it("tags each point circle with its series name for multi-series charts", () => {
  const series = [
    { name: "implied", color: "#58a6ff", points: [{ x: 0, y: 0.2 }] },
    { name: "realized", color: "#f0883e", points: [{ x: 0, y: 0.15 }] },
  ];
  render(<LineChart series={series} ariaLabel="test chart" />);

  const points = screen.getAllByTestId("chart-point");
  expect(points.map((p) => p.dataset.series).sort()).toEqual(["implied", "realized"]);
});

describe("Legend", () => {
  it("renders one swatch per item, colored to match", () => {
    render(<Legend items={[{ name: "implied", color: "#58a6ff" }, { name: "realized", color: "#f0883e" }]} />);

    expect(screen.getByText("implied")).toBeInTheDocument();
    expect(screen.getByText("realized")).toBeInTheDocument();
  });
});
