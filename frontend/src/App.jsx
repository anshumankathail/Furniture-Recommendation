import { useState } from "react";
import { getRecommendations } from "./api";

function App() {
  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    if (!query && !imageFile) {
      alert("Please provide either text or an image.");
      return;
    }
    if (query && imageFile) {
      alert("Please provide only text OR image, not both.");
      return;
    }

    const matches = await getRecommendations(query, imageFile);
    setResults(matches);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl mb-4">Furniture Recommendation</h1>

      <input
        type="text"
        placeholder="Enter your query"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="border p-2 mr-2"
        disabled={imageFile !== null} // disable if image selected
      />

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImageFile(e.target.files[0])}
        className="mr-2"
        disabled={query.length > 0} // disable if text entered
      />

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
