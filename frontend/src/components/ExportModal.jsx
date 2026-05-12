import { useEffect, useState } from "react";
import { X, Sparkles, AlertCircle, Cpu, Package } from "lucide-react";
import {
  fetchAvailableModels,
  fetchExportQuantizations,
  startExport,
} from "../api/client";

export default function ExportModal({ open, onClose, onSubmitted }) {
  const [ftModels, setFtModels] = useState([]);
  const [quants, setQuants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedFtId, setSelectedFtId] = useState(null);
  const [selectedQuant, setSelectedQuant] = useState("Q4_K_M");
  const [outputName, setOutputName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [models, quantsData] = await Promise.all([
          fetchAvailableModels(),
          fetchExportQuantizations(),
        ]);
        if (cancelled) return;

        const fts = (models || []).filter((m) => m.kind === "ft");
        setFtModels(fts);
        setQuants(quantsData || []);

        // Auto-select primo FT + quant default
        if (fts.length > 0 && selectedFtId == null) {
          setSelectedFtId(fts[0].model_id);
        }
        const defaultQuant = (quantsData || []).find((q) => q.is_default);
        if (defaultQuant) setSelectedQuant(defaultQuant.value);
      } catch (e) {
        if (!cancelled) setError(e.response?.data?.detail || e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open]);

  const handleSubmit = async () => {
    if (!selectedFtId) {
      setSubmitError("Seleziona un modello fine-tuned");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await startExport({
        ft_model_id: selectedFtId,
        quantization: selectedQuant,
        output_name: outputName.trim() || null,
      });
      onSubmitted(result);
      // Reset
      setOutputName("");
      onClose();
    } catch (e) {
      setSubmitError(e.response?.data?.detail || e.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-zinc-800">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Package className="w-5 h-5 text-indigo-400" />
            Esporta come GGUF
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-zinc-800 text-zinc-400"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-5">
          {loading && (
            <div className="text-sm text-zinc-400">Caricamento modelli…</div>
          )}

          {error && (
            <div className="bg-red-950/40 border border-red-900/60 rounded-lg p-3 text-sm text-red-200 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <strong>Errore:</strong> {error}
              </div>
            </div>
          )}

          {!loading && !error && ftModels.length === 0 && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 text-center text-zinc-400 text-sm">
              Nessun modello fine-tuned disponibile.
              <br />
              Esegui un training per poterlo esportare.
            </div>
          )}

          {!loading && !error && ftModels.length > 0 && (
            <>
              {/* FT model picker */}
              <div>
                <label className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1.5 block">
                  Modello fine-tuned
                </label>
                <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                  {ftModels.map((m) => {
                    const loss = m.metadata?.final_loss;
                    const steps = m.metadata?.total_steps;
                    const selected = m.model_id === selectedFtId;
                    return (
                      <button
                        key={m.key}
                        onClick={() => setSelectedFtId(m.model_id)}
                        className={`w-full text-left p-2.5 rounded-lg border transition-colors ${
                          selected
                            ? "border-indigo-500 bg-indigo-500/10"
                            : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <Sparkles
                            className={`w-3.5 h-3.5 flex-shrink-0 ${
                              selected ? "text-indigo-400" : "text-zinc-500"
                            }`}
                          />
                          <span className="text-sm text-zinc-100 truncate flex-1">
                            {m.display_name}
                          </span>
                        </div>
                        <div className="text-[11px] text-zinc-500 mt-0.5 flex items-center gap-3 ml-5">
                          {m.base_model_name && <span>← {m.base_model_name}</span>}
                          {loss != null && (
                            <span>
                              loss{" "}
                              <strong className="text-zinc-300 font-mono">
                                {loss.toFixed(3)}
                              </strong>
                            </span>
                          )}
                          {steps != null && <span>{steps} step</span>}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Output name */}
              <div>
                <label className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1.5 block">
                  Nome file (opzionale)
                </label>
                <input
                  type="text"
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  placeholder="Auto-generato se vuoto"
                  maxLength={128}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
                />
                <div className="text-[11px] text-zinc-500 mt-1">
                  Il file finale sarà{" "}
                  <code className="text-zinc-400">
                    {(outputName.trim() || "<auto>") + "__" + selectedQuant + ".gguf"}
                  </code>
                </div>
              </div>

              {/* Quantization */}
              <div>
                <label className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1.5 block">
                  Quantizzazione
                </label>
                <div className="space-y-1.5">
                  {quants.map((q) => {
                    const selected = q.value === selectedQuant;
                    return (
                      <button
                        key={q.value}
                        onClick={() => setSelectedQuant(q.value)}
                        className={`w-full text-left p-2.5 rounded-lg border transition-colors ${
                          selected
                            ? "border-indigo-500 bg-indigo-500/10"
                            : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                              selected
                                ? "border-indigo-400 bg-indigo-400"
                                : "border-zinc-600"
                            }`}
                          />
                          <span className="text-sm text-zinc-100 font-medium">
                            {q.label}
                          </span>
                          {q.is_default && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded border border-emerald-700 text-emerald-300 bg-emerald-900/20">
                              default
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-zinc-500 mt-0.5 ml-5">
                          {q.description}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {submitError && (
            <div className="bg-red-950/40 border border-red-900/60 rounded-lg p-3 text-sm text-red-200 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>{submitError}</div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-zinc-800">
          <button
            onClick={onClose}
            disabled={submitting}
            className="px-4 py-2 text-sm rounded-lg border border-zinc-800 hover:border-zinc-700 text-zinc-300"
          >
            Annulla
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || loading || ftModels.length === 0}
            className="px-4 py-2 text-sm rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-medium flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            {submitting ? "Avvio…" : "Esporta"}
          </button>
        </div>
      </div>
    </div>
  );
}