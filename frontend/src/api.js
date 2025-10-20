export async function getRecommendations(query = null, imageFile = null, top_k = 5) {
  const formData = new FormData();

  if (query) {
    formData.append("query", query);
  } else if (imageFile) {
    formData.append("image", imageFile);
  }

  formData.append("top_k", top_k);

  const response = await fetch("http://localhost:8000/recommend", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error: ${text}`);
  }

  const data = await response.json();
  return data.matches || [];
}
