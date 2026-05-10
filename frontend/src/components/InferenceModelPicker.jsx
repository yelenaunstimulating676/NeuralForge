import { useEffect, useState } from "react";
import { Cpu, Sparkles, CheckCircle2, Clock } from "lucide-react";

function ModelOption({ model, selected, onClick }) {
  const isFt = model.kind === "ft";
  const finalLoss = model.metadata?.final_loss;
  const totalSteps = model.metadata?.total_steps;

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-colors ${
        selected
          ? "border-indigo-500 bg-indigo-500/10"
          : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-1">
        <div className="flex items-center gap-1.5 min-w-0 flex-1">
          {isFt ? (
            <Sparkles className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
          ) : (
            <Cpu className="w-3.5 h-3.5 text-zinc-400 flex-shrink-0" />
          )}
          <span className="text-sm font-medium text-zinc-100 truncate">
            {model.display_name}
          </span>
        </div>
        {model.is_loaded && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded border border-emerald-700 text-emerald-300 bg-emerald-900/20 flex items-center gap-1 flex-shrink-0"
            title="Già caricato in VRAM (risposta istantanea)"
          >
            <CheckCircle2 className="w-3 h-3" />
            in cache
          </span>
        )}
      </div>
      <div className="text-[11px] text-zinc-500 flex flex-wrap items-center gap-x-3 gap-y-0.5">
        <span>{isFt ? "Fine-tuned" : "Base"}</span>
        {isFt && model.base_model_name && (
          <span>← {model.base_model_name}</span>
        )}
        {finalLoss != null && (
          <span>
            loss <strong className="text-zinc-300 font-mono">{finalLoss.toFixed(3)}</strong>
          </span>
        )}
        {totalSteps != null && <span>{totalSteps} step</span>}
      </div>
    </button>
  );
}

export default function InferenceModelPicker({
  models,
  selectedKey,
  onSelect,
  label,
  disabled = false,
}) {
  const [open, setOpen] = useState(false);
  const selected = models.find((m) => m.key === selectedKey) || null;

  // Click outside to close
  useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (!e.target.closest(".model-picker-container")) {
        setOpen(false);
      }
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [open]);

  return (
    <div className="model-picker-container relative">
      {label && (
        <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
          {label}
        </div>
      )}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(!open)}
        className={`w-full p-2.5 rounded-lg border transition-colors text-left ${
          disabled
            ? "border-zinc-800 bg-zinc-900 opacity-50 cursor-not-allowed"
            : open
            ? "border-indigo-500 bg-indigo-500/5"
            : "border-zinc-800 hover:border-zinc-700 bg-zinc-900"
        }`}
      >
        {selected ? (
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0 flex-1">
              {selected.kind === "ft" ? (
                <Sparkles className="w-4 h-4 text-indigo-400 flex-shrink-0" />
              ) : (
                <Cpu className="w-4 h-4 text-zinc-400 flex-shrink-0" />
              )}
              <div className="min-w-0 flex-1">
                <div className="text-sm text-zinc-100 truncate">
                  {selected.display_name}
                </div>
                <div className="text-[10px] text-zinc-500">
                  {selected.kind === "ft"
                    ? `FT · loss ${selected.metadata?.final_loss?.toFixed(3) || "?"}`
                    : "Base model"}
                </div>
              </div>
            </div>
            {selected.is_loaded && (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" title="In cache" />
            )}
            <span className="text-zinc-500 text-xs">▾</span>
          </div>
        ) : (
          <span className="text-sm text-zinc-500">Seleziona modello…</span>
        )}
      </button>

      {open && (
        <div className="absolute z-20 mt-1 w-full bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl max-h-96 overflow-y-auto p-1.5 space-y-1.5">
          {models.length === 0 ? (
            <div className="text-xs text-zinc-500 p-3 text-center">
              Nessun modello disponibile.
            </div>
          ) : (
            models.map((m) => (
              <ModelOption
                key={m.key}
                model={m}
                selected={m.key === selectedKey}
                onClick={() => {
                  onSelect(m);
                  setOpen(false);
                }}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}