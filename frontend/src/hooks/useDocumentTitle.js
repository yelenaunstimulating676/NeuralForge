import { useEffect } from "react";

export default function useDocumentTitle(title) {
  useEffect(() => {
    const full = title ? `${title} — NeuralForge` : "NeuralForge";
    document.title = full;
  }, [title]);
}