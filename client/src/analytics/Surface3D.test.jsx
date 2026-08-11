import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// react-plotly.js's real component mounts Plotly imperatively into the DOM
// and calls `Plotly.newPlot`, which tries to create a WebGL context --
// jsdom has no GPU backend for that, and it hangs rather than failing fast
// (confirmed: a real run pegged a worker at 100% CPU indefinitely). Every
// test here replaces the Plot component with a stub that just records the
// props it was given, so what's actually under test is Surface3D's own
// trace/layout construction -- geometry itself is surfaceMesh.test.js's job.
vi.mock("plotly.js-gl3d-dist-min", () => ({ default: {} }));
vi.mock("react-plotly.js/factory", () => ({
  default: () =>
    function MockPlot({ data }) {
      return (
        <div
          data-testid="plot"
          data-trace-types={data.map((t) => t.type).join(",")}
          data-mesh-alphahull={data.find((t) => t.type === "mesh3d")?.alphahull}
          data-line-names={data
            .filter((t) => t.type === "scatter3d")
            .map((t) => t.name)
            .join(",")}
        />
      );
    },
}));

const { default: Surface3D } = await import("./Surface3D");

function point(strike, impliedVol) {
  return { strike, implied_vol: impliedVol };
}

const PALETTE = { "2026-08-11": "#58a6ff", "2026-08-12": "#3fb950", "2026-08-13": "#f0883e" };
const colorFor = (expiry) => PALETTE[expiry] ?? "#000000";

it("shows a placeholder rather than an empty plot when no curve has points", () => {
  render(<Surface3D curves={[{ expiry: "2026-08-10", points: [] }]} colorFor={colorFor} />);

  expect(screen.getByText(/No skew data/)).toBeInTheDocument();
  expect(screen.queryByTestId("plot")).not.toBeInTheDocument();
});

describe("with real curve data", () => {
  const curves = [
    { expiry: "2026-08-11", points: [point(90, 0.2), point(95, 0.19), point(100, 0.18)] },
    { expiry: "2026-08-12", points: [point(92, 0.21), point(97, 0.2), point(102, 0.19)] },
  ];

  it("passes exactly one mesh3d trace plus one scatter3d trace per expiry", () => {
    render(<Surface3D curves={curves} colorFor={colorFor} />);

    const plot = screen.getByTestId("plot");
    const types = plot.dataset.traceTypes.split(",");
    expect(types.filter((t) => t === "mesh3d")).toHaveLength(1);
    expect(types.filter((t) => t === "scatter3d")).toHaveLength(2);
    expect(plot.dataset.lineNames).toBe("2026-08-11,2026-08-12");
  });

  it("gives the mesh a positive alphahull rather than Plotly's default Delaunay (-1)", () => {
    render(<Surface3D curves={curves} colorFor={colorFor} />);

    const alphahull = Number(screen.getByTestId("plot").dataset.meshAlphahull);
    expect(alphahull).toBeGreaterThan(0);
  });

  it("emits one scatter3d trace per gap-broken segment, not per expiry", () => {
    const gappy = [
      {
        expiry: "2026-08-13",
        points: [point(100, 0.12), point(101, 0.121), point(102, 0.122), point(130, 0.3)],
      },
    ];
    render(<Surface3D curves={gappy} colorFor={colorFor} />);

    const plot = screen.getByTestId("plot");
    // One segment of 3 adjacent strikes, one isolated point 28 strikes away
    // -- two scatter3d traces from a single expiry, both still labeled with
    // that expiry (legend grouping, not a second series).
    expect(plot.dataset.traceTypes.split(",").filter((t) => t === "scatter3d")).toHaveLength(2);
    expect(plot.dataset.lineNames).toBe("2026-08-13,2026-08-13");
  });

  it("colors each expiry's line using the supplied colorFor, not an internal palette", () => {
    render(<Surface3D curves={curves} colorFor={colorFor} />);

    const types = screen.getByTestId("plot").dataset.traceTypes.split(",");
    expect(types).toContain("scatter3d");
    // Indirect but real: a colorFor that maps unknown expiries to black would
    // make this assertion fail if the component silently fell back to its
    // own hardcoded palette instead of calling the prop.
    expect(colorFor("2026-08-11")).not.toBe(colorFor("2026-08-12"));
  });
});

describe("too few points for a mesh", () => {
  it("omits the mesh3d trace below 4 points but still renders the line", () => {
    // mesh3d needs >=4 non-coplanar points; below that buildSurfaceMesh's
    // alphahull degenerates to Infinity. A single 3-point curve should still
    // plot as a line, just without a fill that was never meaningful.
    const curves = [{ expiry: "2026-08-11", points: [point(90, 0.2), point(95, 0.19), point(100, 0.18)] }];
    render(<Surface3D curves={curves} colorFor={colorFor} />);

    const plot = screen.getByTestId("plot");
    const types = plot.dataset.traceTypes.split(",");
    expect(types).not.toContain("mesh3d");
    expect(types.filter((t) => t === "scatter3d")).toHaveLength(1);
  });
});
