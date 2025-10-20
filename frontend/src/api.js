const DEPLOY_URL = 'https://furniture-backend-fp9f.onrender.com'
const LOCAL_HOST = 'http://127.0.0.1:8000'

const API_URL = DEPLOY_URL; // change this to deploy url to deploy on a server

export async function getRecommendations(query = null, imageFile = null, top_k = 5) {
  const formData = new FormData();

  if (query) formData.append("query", query);
  if (imageFile) formData.append("image", imageFile);
  formData.append("top_k", top_k);

  const response = await fetch(`${API_URL}/recommend`, {
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
