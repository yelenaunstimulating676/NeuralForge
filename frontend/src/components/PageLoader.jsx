import { Loader2 } from "lucide-react";

export default function PageLoader({ message = "Caricamento..." }) {
  return (
    <div className="flex items-center gap-2 text-zinc-400 p-8">
      <Loader2 size={18} className="animate-spin" />
      <span className="text-sm">{message}</span>
    </div>
  );
}