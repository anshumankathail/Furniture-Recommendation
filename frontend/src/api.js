import axios from "axios";

export const getRecommendations = async (query, imageFile) => {
  try {
    const formData = new FormData();

    if (query) formData.append("query", query);
    if (imageFile) formData.append("image", imageFile);
    formData.append("top_k", 5);

    const response = await axios.post("http://127.0.0.1:8000/recommend", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data.matches;
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return [];
  }
};
