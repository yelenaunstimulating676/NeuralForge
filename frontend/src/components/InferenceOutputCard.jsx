import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Hash,
  Activity,
  AlertCircle,
} from "lucide-react";

function FinishBadge({ reason }) {
  if (!reason) return null;
  const map = {
    eos: { color: "text-emerald-300 bg-emerald-900/20 border-emerald-700/50", label: "completata" },
    length: { color: "text-amber-300 bg-amber-900/20 border-amber-700/50", label: "limite token" },
    unknown: { color: "text-zinc-400 bg-zinc-900/20 border-zinc-700", label: "interrotta" },
  };
  const c = map[reason] || map.unknown;
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${c.color}`}>
      {c.label}
    </span>
  );
}

export default function InferenceOutputCard({
  modelName,
  modelKind,
  state,        // 'idle' | 'loading' | 'success' | 'error'
  result,       // GenerationResult quando success
  error,        // string quando error
  loadingHint,  // 'loading_model' | 'generating' | null
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 flex flex-col min-h-[240px]">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-3 pb-3 border-b border-zinc-800">
        <div className="min-w-0 flex-1">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
            {modelKind === "ft" ? "Fine-tuned" : "Base model"}
          </div>
          <div className="text-sm font-medium text-zinc-100 truncate">
            {modelName || "-"}
          </div>
        </div>
        {state === "success" && result && (
          <FinishBadge reason={result.finish_reason} />
        )}
      </div>

      {/* Body */}
      <div className="flex-1 flex flex-col">
        {state === "idle" && (
          <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">
            La risposta apparirà qui dopo il "Genera"
          </div>
        )}

        {state === "loading" && (
          <div className="flex-1 flex flex-col items-center justify-center gap-2 text-zinc-400">
            <Loader2 className="w-6 h-6 animate-spin text-indigo-400" />
            <div className="text-sm">
              {loadingHint === "loading_model"
                ? "Caricamento modello in VRAM…"
                : "Generazione in corso…"}
            </div>
            {loadingHint === "loading_model" && (
              <div className="text-xs text-zinc-600">
                Prima volta: ~5-10s. Poi sarà istantaneo.
              </div>
            )}
          </div>
        )}

        {state === "error" && (
          <div className="flex-1 flex flex-col items-start gap-2 bg-red-950/30 border border-red-900/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-red-300">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm font-medium">Errore</span>
            </div>
            <div className="text-xs text-red-200">{error}</div>
          </div>
        )}

        {state === "success" && result && (
          <>
            <pre className="flex-1 text-sm text-zinc-100 whitespace-pre-wrap font-sans leading-relaxed">
              {result.text || <span className="text-zinc-600 italic">(risposta vuota)</span>}
            </pre>
            <div className="mt-3 pt-3 border-t border-zinc-800 flex items-center gap-3 text-[11px] text-zinc-500 flex-wrap">
              <span className="flex items-center gap-1">
                <Hash className="w-3 h-3" /> {result.tokens_generated} token
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> {result.elapsed_seconds.toFixed(1)}s
              </span>
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3" /> {result.throughput_tokens_per_sec.toFixed(0)} tok/s
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}