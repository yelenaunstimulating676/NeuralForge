import { useEffect, useMemo, useState } from "react";
import { MessageSquare, Sparkles, Trash2, AlertCircle } from "lucide-react";
import {
  fetchAvailableModels,
  generateInference,
  unloadAllModels,
} from "../api/client";
import InferenceModelPicker from "../components/InferenceModelPicker";
import SamplingControls from "../components/SamplingControls";
import InferenceOutputCard from "../components/InferenceOutputCard";
import useDocumentTitle from "../hooks/useDocumentTitle";
import PageLoader from "../components/PageLoader";
import { useNavigate } from "react-router-dom";

const DEFAULT_PARAMS = {
  max_new_tokens: 256,
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  repetition_penalty: 1.1,
  do_sample: true,
};

const SUGGESTED_PROMPTS = [
  "Qual è la capitale d'Italia?",
  "Qual è la capitale della Francia?",
  "Spiegami la fotosintesi in 3 frasi",
];

export default function Inference() {
  useDocumentTitle("Inference");
  const [models, setModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [errorModels, setErrorModels] = useState(null);

  const [prompt, setPrompt] = useState("");
  const [leftKey, setLeftKey] = useState(null);
  const [rightKey, setRightKey] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);

  // Stati per ciascuna colonna: { state, result, error, loadingHint }
  const [leftOut, setLeftOut] = useState({ state: "idle" });
  const [rightOut, setRightOut] = useState({ state: "idle" });

  const [generating, setGenerating] = useState(false);
  const navigate = useNavigate();

  // Carica lista modelli all'avvio
  const refreshModels = async () => {
    try {
      const data = await fetchAvailableModels();
      setModels(data || []);
      setErrorModels(null);
      // Auto-seleziona se vuoto: primo base + primo ft
      if (!leftKey && !rightKey && data && data.length > 0) {
        const firstBase = data.find((m) => m.kind === "base");
        const firstFt = data.find((m) => m.kind === "ft");
        if (firstBase) setLeftKey(firstBase.key);
        if (firstFt) setRightKey(firstFt.key);
      }
    } catch (e) {
      console.error("Errore caricamento modelli:", e);
      setErrorModels(e.response?.data?.detail || e.message);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    refreshModels();
  }, []);

  const leftModel = useMemo(
    () => models.find((m) => m.key === leftKey),
    [models, leftKey]
  );
  const rightModel = useMemo(
    () => models.find((m) => m.key === rightKey),
    [models, rightKey]
  );

  const runOne = async (model, setOut) => {
    if (!model) {
      setOut({ state: "idle" });
      return;
    }
    const hint = model.is_loaded ? "generating" : "loading_model";
    setOut({ state: "loading", loadingHint: hint });

    try {
      const result = await generateInference({
        prompt,
        model_kind: model.kind,
        model_id: model.model_id,
        params,
      });
      setOut({ state: "success", result });
    } catch (e) {
      console.error("Generation error:", e);
      const msg = e.response?.data?.detail || e.message || "Errore sconosciuto";
      setOut({ state: "error", error: typeof msg === "string" ? msg : JSON.stringify(msg) });
    }
  };

  const onGenerate = async () => {
    if (!prompt.trim()) {
      alert("Scrivi un prompt prima di generare.");
      return;
    }
    if (!leftModel && !rightModel) {
      alert("Seleziona almeno un modello.");
      return;
    }

    setGenerating(true);
    try {
      // Sequenziale: prima sinistra, poi destra
      await runOne(leftModel, setLeftOut);
      await runOne(rightModel, setRightOut);
      // Refresh dei modelli per aggiornare il flag is_loaded
      await refreshModels();
    } finally {
      setGenerating(false);
    }
  };

  const onClearCache = async () => {
    if (!confirm("Scaricare tutti i modelli dalla VRAM?")) return;
    try {
      await unloadAllModels();
      await refreshModels();
    } catch (e) {
      alert("Errore: " + (e.response?.data?.detail || e.message));
    }
  };

  if (loadingModels) {
    return <PageLoader message="Caricamento modelli..." />;
  }

  if (errorModels) {
    return (
      <div className="p-8 max-w-2xl">
        <h1 className="text-3xl font-bold mb-2">Inference</h1>
        <div className="bg-red-950/40 border border-red-900/60 rounded-xl p-4 text-sm text-red-200 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <strong>Errore caricamento modelli:</strong> {errorModels}
          </div>
        </div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="p-8">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <MessageSquare className="w-7 h-7 text-indigo-400" />
          Inference
        </h1>
        <div className="bg-zinc-900 border border-zinc-800 border-dashed rounded-2xl p-12 text-center mt-6">
          <Sparkles className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
          <p className="text-zinc-400 mb-1">Nessun modello disponibile.</p>
          <p className="text-xs text-zinc-600 mb-4">
            Scarica un base model e/o esegui un fine-tuning per usare questa pagina.
          </p>
          <button
            onClick={() => navigate("/models")}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg inline-flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            Vai a Models
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-3xl font-bold mb-1 flex items-center gap-2">
            <MessageSquare className="w-7 h-7 text-indigo-400" />
            Inference
          </h1>
          <p className="text-zinc-400">
            Confronta le risposte dei tuoi modelli affiancate. Modelli FT memorizzano il tuo dataset.
          </p>
        </div>
        <button
          onClick={onClearCache}
          className="px-3 py-2 text-xs rounded-lg border border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-zinc-400 flex items-center gap-1.5"
          title="Libera la VRAM"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Svuota cache VRAM
        </button>
      </div>

      {/* Prompt */}
      <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
        <label className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2 block">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Es. Qual è la capitale d'Italia?"
          rows={3}
          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-zinc-100 text-sm resize-y focus:border-indigo-500 focus:outline-none"
        />
        <div className="mt-2 flex flex-wrap gap-1.5">
          {SUGGESTED_PROMPTS.map((s) => (
            <button
              key={s}
              onClick={() => setPrompt(s)}
              className="text-[11px] px-2 py-1 rounded-md bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </section>

      {/* Modelli + Sampling */}
      <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InferenceModelPicker
            label="Modello — colonna sinistra"
            models={models}
            selectedKey={leftKey}
            onSelect={(m) => setLeftKey(m.key)}
            disabled={generating}
          />
          <InferenceModelPicker
            label="Modello — colonna destra"
            models={models}
            selectedKey={rightKey}
            onSelect={(m) => setRightKey(m.key)}
            disabled={generating}
          />
        </div>
        <SamplingControls params={params} onChange={setParams} />
        <button
          onClick={onGenerate}
          disabled={generating || !prompt.trim()}
          className="w-full px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-500 disabled:cursor-not-allowed text-white font-medium flex items-center justify-center gap-2 transition-colors"
        >
          <Sparkles className="w-4 h-4" />
          {generating ? "Generazione in corso…" : "Genera"}
        </button>
      </section>

      {/* Output side-by-side */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InferenceOutputCard
          modelName={leftModel?.display_name}
          modelKind={leftModel?.kind}
          state={leftOut.state}
          result={leftOut.result}
          error={leftOut.error}
          loadingHint={leftOut.loadingHint}
        />
        <InferenceOutputCard
          modelName={rightModel?.display_name}
          modelKind={rightModel?.kind}
          state={rightOut.state}
          result={rightOut.result}
          error={rightOut.error}
          loadingHint={rightOut.loadingHint}
        />
      </section>
    </div>
  );
}