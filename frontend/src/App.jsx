import { useState } from "react";
import { getRecommendations } from "./api";

function App() {
  const [mode, setMode] = useState("text"); // "text" or "image"
  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    const matches = await getRecommendations(mode === "text" ? query : null, mode === "image" ? imageFile : null);
    setResults(matches);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl mb-4">Furniture Recommendation</h1>

      {/* Mode selection */}
      <div className="mb-4">
        <label className="mr-4">
          <input
            type="radio"
            value="text"
            checked={mode === "text"}
            onChange={() => setMode("text")}
            className="mr-1"
          />
          Text Search
        </label>
        <label>
          <input
            type="radio"
            value="image"
            checked={mode === "image"}
            onChange={() => setMode("image")}
            className="mr-1"
          />
          Image Search
        </label>
      </div>

      {/* Conditional input */}
      {mode === "text" ? (
        <input
          type="text"
          placeholder="Enter your query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="border p-2 mr-2"
        />
      ) : (
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImageFile(e.target.files[0])}
          className="mr-2"
        />
      )}

      <button
        onClick={handleSearch}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Search
      </button>

      <div className="mt-6">
        {results.length === 0 && <p>No results yet.</p>}
        {results.map((item, idx) => (
          <div key={idx} className="border p-4 mb-2">
            <p><strong>{item.title}</strong></p>
            <p>{item.brand}</p>
            <img src={item.image_url} alt={item.title} className="w-32 h-32" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
