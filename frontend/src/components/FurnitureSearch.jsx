import React, { useState } from "react";
import { getRecommendations } from "./api"; // import the function

export default function FurnitureSearch() {
  const [query, setQuery] = useState("");
  const [image, setImage] = useState(null);
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const matches = await getRecommendations(query, image, topK);
      setResults(matches);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Furniture Search</h2>
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          placeholder="Enter furniture description..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="border p-2 mr-2"
        />
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          className="mr-2"
        />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded">
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      <div>
        {results.length > 0 && <h3 className="font-semibold mb-2">Results:</h3>}
        <ul>
          {results.map((item, idx) => (
            <li key={idx} className="mb-2">
              {item.title} - {item.brand} - {item.color}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
