// import Plotly from "plotly.js-dist-min";
// import { useEffect, useRef, useState } from "react";
// import { io } from "socket.io-client";

// const socket = io("http://127.0.0.1:5000");

// function Visualiser() {
//   const plotRef = useRef(null);
//   const [surface, setSurface] = useState(null);
//   const [trajectory, setTrajectory] = useState([]);

//   useEffect(() => {
//     socket.on("surface", (data) => {
//       setSurface(data);
//       setTrajectory([]);
//     });
//     socket.on("update", (data) => {
//       console.log("update received", data);
//       setTrajectory((prev) => [...prev, [data.w1, data.w2, data.loss]]);
//     });
//     return () => {
//       socket.off("surface");
//       socket.off("update");
//     };
//   }, []);

//   useEffect(() => {
//     if (!surface || !plotRef.current) return;

//     const trajX = trajectory.map((p) => p[0]);
//     const trajY = trajectory.map((p) => p[1]);

//     Plotly.react(
//       plotRef.current,
//       [
//         {
//           z: surface.loss,
//           x: surface.w1,
//           y: surface.w2,
//           type: "contour",
//           colorscale: "Viridis",
//           contours: { showlabels: true },
//         },
//         {
//           x: trajX,
//           y: trajY,
//           mode: "lines+markers",
//           type: "scatter",
//           name: "Optimizer Path",
//           line: { color: "red" },
//           marker: { size: 6 },
//         },
//       ],
//       {
//         title: "Loss Surface & Optimization Trajectory",
//         xaxis: { title: "w1", range: [-2, 2] },
//         yaxis: { title: "w2", range: [-2, 2] },
//         autosize: true,
//       },
//     );
//   }, [surface, trajectory]);

//   if (!surface) {
//     return (
//       <div className="text-gray-500 text-center mt-10">
//         Waiting for surface...
//       </div>
//     );
//   }

//   return <div ref={plotRef} style={{ width: "100%", height: "600px" }} />;
// }

// export default Visualiser;
import Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://127.0.0.1:5000");

function Visualiser() {
  const plotRef = useRef(null);
  const lossRef = useRef(null);
  const [surface, setSurface] = useState(null);
  const [trajectory, setTrajectory] = useState([]);

  useEffect(() => {
    socket.on("surface", (data) => {
      setSurface(data);
      setTrajectory([]);
    });
    socket.on("update", (data) => {
      if (Math.abs(data.w1) <= 2 && Math.abs(data.w2) <= 2) {
        setTrajectory((prev) => [...prev, [data.w1, data.w2, data.loss]]);
      }
    });
    return () => {
      socket.off("surface");
      socket.off("update");
    };
  }, []);

  // Contour + trajectory
  useEffect(() => {
    if (!surface || !plotRef.current) return;
    const trajX = trajectory.map((p) => p[0]);
    const trajY = trajectory.map((p) => p[1]);

    Plotly.react(
      plotRef.current,
      [
        {
          z: surface.loss,
          x: surface.w1,
          y: surface.w2,
          type: "contour",
          colorscale: "Viridis",
          contours: { showlabels: true },
        },
        {
          x: trajX,
          y: trajY,
          mode: "lines+markers",
          type: "scatter",
          name: "Optimizer Path",
          line: { color: "red", width: 2 },
          marker: { size: 6, color: "red" },
        },
        // Highlight current position
        {
          x: trajX.slice(-1),
          y: trajY.slice(-1),
          mode: "markers",
          type: "scatter",
          name: "Current",
          marker: { size: 14, color: "white", symbol: "star" },
        },
      ],
      {
        title: "Loss Surface & Optimization Trajectory",
        xaxis: { title: "w1", range: [-2, 2] },
        yaxis: { title: "w2", range: [-2, 2] },
        autosize: true,
      },
    );
  }, [surface, trajectory]);

  // Loss curve
  useEffect(() => {
    if (!lossRef.current || trajectory.length === 0) return;
    const steps = trajectory.map((_, i) => i * 10);
    const losses = trajectory.map((p) => p[2]);

    Plotly.react(
      lossRef.current,
      [
        {
          x: steps,
          y: losses,
          type: "scatter",
          mode: "lines",
          name: "Loss",
          line: { color: "#6366f1", width: 2 },
        },
      ],
      {
        title: "Loss Curve",
        xaxis: { title: "Epoch" },
        yaxis: { title: "Loss" },
        autosize: true,
        margin: { t: 40, b: 40, l: 50, r: 20 },
      },
    );
  }, [trajectory]);

  const latest = trajectory[trajectory.length - 1];

  if (!surface) {
    return (
      <div className="text-gray-400 text-center mt-10 text-sm">
        ⏳ Waiting for training to start...
      </div>
    );
  }

  return (
    <div>
      {/* Live weight & loss display */}
      <div className="grid grid-cols-3 gap-4 mt-6 mb-4">
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center">
          <p className="text-xs text-blue-400 uppercase font-semibold mb-1">
            w1
          </p>
          <p className="text-xl font-mono font-bold text-blue-700">
            {latest ? latest[0].toFixed(4) : "—"}
          </p>
        </div>
        <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 text-center">
          <p className="text-xs text-purple-400 uppercase font-semibold mb-1">
            w2
          </p>
          <p className="text-xl font-mono font-bold text-purple-700">
            {latest ? latest[1].toFixed(4) : "—"}
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
          <p className="text-xs text-green-400 uppercase font-semibold mb-1">
            Loss
          </p>
          <p className="text-xl font-mono font-bold text-green-700">
            {latest ? latest[2].toFixed(6) : "—"}
          </p>
        </div>
      </div>

      <p className="text-center text-sm text-gray-400 mb-4">
        Epoch {trajectory.length * 10} — {trajectory.length} updates
      </p>

      {/* Loss curve */}
      <div ref={lossRef} style={{ width: "100%", height: "220px" }} />

      {/* Contour + trajectory */}
      <div ref={plotRef} style={{ width: "100%", height: "500px" }} />
    </div>
  );
}

export default Visualiser;
