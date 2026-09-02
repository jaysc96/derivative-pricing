import { useMemo } from "react";
// plotly.js-gl3d-dist-min ships only the cartesian + gl3d (scatter3d,
// mesh3d, ...) trace families this view needs -- the full `plotly.js`
// package bundles every trace type (2D, geo, mapbox, finance) for a chart
// that uses exactly two. react-plotly.js's default export hardcodes the
// full package; the /factory entry point takes any Plotly-compatible module
// instead, which is what makes using the slim build possible.
import Plotly from "plotly.js-gl3d-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { buildSurfaceMesh } from "./surfaceMesh";

const Plot = createPlotlyComponent(Plotly);

const MESH_COLOR = "#58a6ff";

//: mesh3d needs at least 4 non-coplanar points to fill any facet at all.
//: Below that, buildSurfaceMesh's alpha-shape radius (a nearest-neighbor
//: distance with nothing to be nearest to) degenerates to Infinity -- so
//: this is also the same floor a meaningful alpha-shape radius needs.
const MIN_MESH_POINTS = 4;

function axisLayout(title, ticks) {
  return {
    title: { text: title, font: { color: "#8b93a3", size: 11 } },
    tickvals: ticks.map((t) => t.value),
    ticktext: ticks.map((t) => t.label),
    tickfont: { color: "#5f6773", size: 9 },
    gridcolor: "#2f3745",
    zerolinecolor: "#3d4759",
    // An explicit opaque fill, not a translucent white over a transparent
    // paper -- confirmed empirically that the alpha did not composite the
    // way it does for 2D traces, rendering as solid white instead of a
    // faint dark tint and making the theme's own tick-label colors
    // (tuned for a dark pane) unreadable against it.
    showbackground: true,
    backgroundcolor: "#12151b",
  };
}

export default function Surface3D({ curves, colorFor }) {
  const geometry = useMemo(() => buildSurfaceMesh(curves), [curves]);

  const figure = useMemo(() => {
    if (!geometry) return null;
    const { lines, mesh, axisTicks } = geometry;

    const lineTraces = lines.flatMap((line) =>
      line.segments.map((segment, segmentIndex) => ({
        type: "scatter3d",
        mode: "lines+markers",
        name: line.expiry,
        legendgroup: line.expiry,
        showlegend: segmentIndex === 0,
        x: segment.map((p) => p.x),
        y: segment.map((p) => p.y),
        z: segment.map((p) => p.z),
        line: { color: colorFor(line.expiry), width: 4 },
        marker: { size: 2.5, color: colorFor(line.expiry) },
        customdata: segment.map((p) => [p.strike, p.impliedVol]),
        hovertemplate: `${line.expiry}<br>strike %{customdata[0]}<br>vol %{customdata[1]:.2%}<extra></extra>`,
      })),
    );

    // Below MIN_MESH_POINTS there's nothing for an alpha-shape to fill (and
    // buildSurfaceMesh's own alphahull would be Infinity) -- the per-expiry
    // lines above still render, just without a connecting sheet, which is
    // the honest answer when there isn't enough data for one.
    const meshTrace =
      mesh.x.length >= MIN_MESH_POINTS
        ? [
            {
              type: "mesh3d",
              x: mesh.x,
              y: mesh.y,
              z: mesh.z,
              alphahull: mesh.alphahull,
              opacity: 0.35,
              color: MESH_COLOR,
              showscale: false,
              hoverinfo: "skip",
              flatshading: true,
            },
          ]
        : [];

    const layout = {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: axisLayout("Strike", axisTicks.x),
        yaxis: axisLayout("Expiry", axisTicks.y),
        zaxis: axisLayout("Implied vol", axisTicks.z),
        camera: { eye: { x: 1.7, y: -1.7, z: 0.9 } },
        aspectmode: "cube",
      },
      margin: { l: 0, r: 0, t: 10, b: 0 },
      showlegend: true,
      legend: { font: { color: "#8b93a3", size: 11 }, bgcolor: "rgba(0,0,0,0)", x: 0, y: 1 },
    };

    return { data: [...meshTrace, ...lineTraces], layout };
  }, [geometry, colorFor]);

  if (!figure) {
    return <div className="placeholder-panel">No skew data to build a surface from yet.</div>;
  }

  return (
    <Plot
      data={figure.data}
      layout={figure.layout}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: "100%", height: "480px" }}
      useResizeHandler
    />
  );
}
