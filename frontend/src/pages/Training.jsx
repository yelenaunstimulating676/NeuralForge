import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Cpu,
  Database,
  Settings,
  Sparkles,
  Zap,
  Tag,
  AlertTriangle,
  Clock,
  Hash,
  BarChart3,
} from "lucide-react";
import {
  fetchBaseModels,
  fetchDatasets,
  estimateTraining,
  startTraining,
} from "../api/client";
import TrainingModelPicker from "../components/TrainingModelPicker";
import TrainingDatasetPicker from "../components/TrainingDatasetPicker";
import TrainingConfigForm from "../components/TrainingConfigForm";
import useDocumentTitle from "../hooks/useDocumentTitle";
import PageLoader from "../components/PageLoader";

const DEFAULT_CONFIG = {
  num_epochs: 3,
  per_device_batch_size: 2,
  grad_accum_steps: 2,
  learning_rate: 2e-4,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  max_seq_length: 1024,
  warmup_ratio: 0.03,
  weight_decay: 0.01,
  use_4bit: true,
  use_8bit_optimizer: true,
  log_every_n_steps: 1,
  save_every_n_steps: 0,
  keep_last_n: 3,
  train_on_response_only: true,
  compute_dtype: "bfloat16",
  max_grad_norm: 1.0,
  min_lr_ratio: 0.0,
  max_steps: 0,
};

function formatTime(seconds) {
  if (!seconds || seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h}h ${mm}m` : `${h}h`;
}

function formatNumber(n) {
  if (n == null) return "-";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

function StepBadge({ n }) {
  return (
    <span className="flex-shrink-0 w-6 h-6 rounded-md bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center text-xs font-bold text-indigo-300">
      {n}
    </span>
  );
}

function StimaBox({ icon: Icon, label, value, sublabel, danger }) {
  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
        <Icon className="w-3 h-3" />
        {label}
      </div>
      <div
        className={`text-base font-semibold ${
          danger ? "text-red-400" : "text-zinc-100"
        }`}
      >
        {value}
      </div>
      {sublabel && (
        <div className="text-[10px] text-zinc-500">{sublabel}</div>
      )}
    </div>
  );
}

export default function Training() {
  useDocumentTitle("Training");
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);

  const [selectedModelId, setSelectedModelId] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [finetunedName, setFinetunedName] = useState("");
  const [config, setConfig] = useState(DEFAULT_CONFIG);

  const [estimation, setEstimation] = useState(null);
  const [estimating, setEstimating] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [modelsRes, datasetsRes] = await Promise.all([
          fetchBaseModels(),
          fetchDatasets(),
        ]);
        if (cancelled) return;
        setModels(modelsRes || []);
        setDatasets(datasetsRes || []);
        if (modelsRes?.length === 1) setSelectedModelId(modelsRes[0].id);
        if (datasetsRes?.length === 1) setSelectedDatasetId(datasetsRes[0].id);
      } catch (e) {
        console.error("Errore caricamento risorse:", e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedModelId || !selectedDatasetId) {
      setEstimation(null);
      return;
    }
    setEstimating(true);
    const handle = setTimeout(async () => {
      try {
        const data = await estimateTraining({
          base_model_id: selectedModelId,
          dataset_id: selectedDatasetId,
          num_epochs: config.num_epochs,
          per_device_batch_size: config.per_device_batch_size,
          grad_accum_steps: config.grad_accum_steps,
          max_seq_length: config.max_seq_length,
          lora_r: config.lora_r,
          use_4bit: config.use_4bit,
        });
        setEstimation(data);
      } catch (e) {
        console.error("Errore estimation:", e);
        setEstimation(null);
      } finally {
        setEstimating(false);
      }
    }, 400);
    return () => clearTimeout(handle);
  }, [
    selectedModelId,
    selectedDatasetId,
    config.num_epochs,
    config.per_device_batch_size,
    config.grad_accum_steps,
    config.max_seq_length,
    config.lora_r,
    config.use_4bit,
  ]);

  const canSubmit = useMemo(
    () =>
      selectedModelId && selectedDatasetId && !submitting && !estimating,
    [selectedModelId, selectedDatasetId, submitting, estimating]
  );

  const onSubmit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const data = await startTraining({
        base_model_id: selectedModelId,
        dataset_id: selectedDatasetId,
        finetuned_name: finetunedName.trim() || null,
        ...config,
      });
      navigate(`/training/live/${data.run_id}`, {
        state: { jobId: data.job_id },
      });
    } catch (e) {
      console.error("Errore start:", e);
      const detail = e.response?.data?.detail || e.message || String(e);
      setSubmitError(detail);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <PageLoader message="Caricamento risorse..." />;
  }

  if (!models.length || !datasets.length) {
    return (
      <div className="p-8 max-w-2xl">
        <h1 className="text-3xl font-bold mb-2">Training</h1>
        <p className="text-zinc-400 mb-6">
          Fine-tuna un modello base con il tuo dataset.
        </p>
        <div className="bg-amber-950/30 border border-amber-900/50 rounded-2xl p-6">
          <p className="text-amber-200 font-medium mb-2">
            Mancano risorse per avviare un training.
          </p>
          <ul className="text-amber-200/80 text-sm space-y-1 list-disc pl-5">
            {!models.length && (
              <li>
                Scarica almeno un modello dalla pagina <strong>Models</strong>.
              </li>
            )}
            {!datasets.length && (
              <li>
                Crea almeno un dataset dalla pagina <strong>Dataset</strong>.
              </li>
            )}
          </ul>
        </div>
      </div>
    );
  }

  const ready = !!(selectedModelId && selectedDatasetId);
  const vramHigh = estimation && estimation.estimated_vram_mb > 12000;

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-1 flex items-center gap-2">
          <Sparkles className="w-7 h-7 text-indigo-400" />
          Training
        </h1>
        <p className="text-zinc-400">
          Configura e avvia un fine-tuning. La pagina monitor live ti mostrerà
          metriche in tempo reale.
        </p>
      </div>

      {/* Riga 1: Modello base | Dataset (50/50) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <StepBadge n={1} />
            <Cpu className="w-4 h-4 text-zinc-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Modello base</h2>
          </div>
          <TrainingModelPicker
            models={models}
            selectedId={selectedModelId}
            onSelect={setSelectedModelId}
          />
        </section>

        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <StepBadge n={2} />
            <Database className="w-4 h-4 text-zinc-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Dataset</h2>
          </div>
          <TrainingDatasetPicker
            datasets={datasets}
            selectedId={selectedDatasetId}
            onSelect={setSelectedDatasetId}
          />
        </section>
      </div>

      {/* Riga 2: Configurazione (largo) | Stima + Submit (sticky stretto) */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-4">
        {/* Configurazione */}
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <StepBadge n={3} />
            <Settings className="w-4 h-4 text-zinc-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Configurazione</h2>
            <span className="text-xs text-zinc-500 ml-auto">
              I default sono buoni per iniziare
            </span>
          </div>
          <TrainingConfigForm config={config} onChange={setConfig} />
        </section>

        {/* Sticky: stima + nome + submit */}
        <aside className="xl:sticky xl:top-6 xl:self-start space-y-3">
          {/* Stima */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5" /> Stima
              </h3>
              <span className="text-[10px] text-zinc-600 italic">
                ±30%
              </span>
            </div>

            {!ready ? (
              <p className="text-xs text-zinc-500 py-2">
                Seleziona modello e dataset.
              </p>
            ) : estimating || !estimation ? (
              <div className="grid grid-cols-2 gap-2 animate-pulse">
                {[0, 1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="h-14 bg-zinc-950 border border-zinc-800 rounded-lg"
                  />
                ))}
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <StimaBox
                    icon={Cpu}
                    label="VRAM"
                    value={`${estimation.estimated_vram_mb.toLocaleString("it-IT")} MB`}
                    danger={vramHigh}
                  />
                  <StimaBox
                    icon={Clock}
                    label="Tempo"
                    value={formatTime(estimation.estimated_time_seconds)}
                  />
                  <StimaBox
                    icon={Hash}
                    label="Step"
                    value={estimation.total_steps.toLocaleString("it-IT")}
                    sublabel={`${estimation.steps_per_epoch}/epoch`}
                  />
                  <StimaBox
                    icon={Zap}
                    label="LoRA"
                    value={formatNumber(estimation.trainable_params_estimated)}
                    sublabel="params"
                  />
                </div>
                {estimation.notes && estimation.notes.length > 0 && (
                  <div className="space-y-1.5 mt-3">
                    {estimation.notes.map((note, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-1.5 text-[11px] text-amber-200 bg-amber-950/30 border border-amber-900/50 rounded-md px-2 py-1.5"
                      >
                        <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                        <span>{note}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Nome modello */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5" /> Nome modello
            </h3>
            <input
              type="text"
              value={finetunedName}
              onChange={(e) => setFinetunedName(e.target.value)}
              placeholder="es. Capitali Expert v1"
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              maxLength={255}
            />
            <p className="text-[10px] text-zinc-500 mt-1">
              Opzionale. Auto-generato se vuoto.
            </p>
          </div>

          {/* Errore */}
          {submitError && (
            <div className="bg-red-950/40 border border-red-900/60 rounded-xl p-3 text-xs text-red-200">
              <strong className="block mb-1">Errore avvio:</strong>
              {submitError}
            </div>
          )}

          {/* Submit */}
          <button
            disabled={!canSubmit}
            onClick={onSubmit}
            className="w-full px-5 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-500 disabled:cursor-not-allowed text-white font-medium rounded-xl flex items-center justify-center gap-2 transition-colors shadow-lg shadow-indigo-500/10"
          >
            <Zap className="w-5 h-5" />
            {submitting ? "Avvio in corso…" : "Avvia Training"}
          </button>
        </aside>
      </div>
    </div>
  );
}