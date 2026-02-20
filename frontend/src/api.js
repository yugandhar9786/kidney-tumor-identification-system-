import axios from "axios";

// Base URL (can be overridden via .env)
const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

// Axios client
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// 🔹 Predict Kidney Tumor
export const predictKidneyTumor = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/predict", formData);
  return response.data;
};
