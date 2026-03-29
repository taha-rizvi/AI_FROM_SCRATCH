import { useState, useEffect } from "react";
import axios from "axios";
import Visualiser from "./Visualiser";
function App() {
  const [epochs, setEpochs] = useState("");
  const [lr, setLr] = useState("");
  const [optimizer, setOptimizer] = useState("SGD");
  const [error, setError] = useState("");

  function handleSelect(e) {
    setOptimizer(e.target.value);
  }
  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (!epochs || !lr || !optimizer) {
      setError("Epochs, Learning rate and optimizer are required!");
      return;
    }
    try {
      const res = await axios.post("http://127.0.0.1:5000/start", {
        epochs: Number(epochs),
        lr: Number(lr),
        optimizer,
      });
    } catch (error) {
      setError(error.response?.data?.error || "Invalid Input");
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center p-4">
      <div className="bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl p-8 max-w-md w-full">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Enter the Hyperparameters
        </h1>
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            value={epochs}
            placeholder="Epochs"
            onChange={(e) => setEpochs(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="text"
            value={lr}
            placeholder="Learning Rate"
            onChange={(e) => setLr(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <select
            onChange={handleSelect}
            value={optimizer}
            className="w-full p-2 border rounded"
          >
            <option>SGD</option>
            <option>Momentum</option>
            <option>RMSprop</option>
            <option>Adam</option>
            <option>Adagrad</option>
          </select>
          <button
            type="submit"
            className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-2 px-4 rounded-lg font-semibold hover:shadow-lg transition-all duration-200"
          >
            Start Training
          </button>
        </form>
        <Visualiser />
      </div>
    </div>
  );
}

export default App;
