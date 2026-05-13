import { useNavigate } from "react-router-dom";
import {
  CheckCircle2,
  XCircle,
  Loader,
  Ban,
  Clock,
  TrendingUp,
  Hash,
  Cpu,
  Database,
  Eye,
  Trash2,
} from "lucide-react";

function StatusPill({ status }) {
  const map = {
    pending: { color: "bg-zinc-700/50 text-zinc-300 border-zinc-600", label: "In coda", icon: Loader },
    running: {
      color: "bg-indigo-500/15 text-indigo-300 border-indigo-500/40",
      label: "In corso",
      icon: Loader,
      spin: true,
    },
    completed: {
      color: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
      label: "Completato",
      icon: CheckCircle2,
    },
    failed: {
      color: "bg-red-500/15 text-red-300 border-red-500/40",
      label: "Fallito",
      icon: XCircle,
    },
    cancelled: {
      color: "bg-amber-500/15 text-amber-300 border-amber-500/40",
      label: "Cancellato",
      icon: Ban,
    },
  };
  const conf = map[status] || map.pending;
  const Icon = conf.icon;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-medium ${conf.color}`}
    >
      <Icon className={`w-3.5 h-3.5 ${conf.spin ? "animate-spin" : ""}`} />
      {conf.label}
    </span>
  );
}

function formatDate(iso) {
  if (!iso) return "-";
  try {
    const d = new Date(iso);
    return d.toLocaleString("it-IT", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function formatDuration(start, end) {
  if (!start) return "-";
  const startMs = new Date(start).getTime();
  const endMs = end ? new Date(end).getTime() : Date.now();
  const sec = Math.round((endMs - startMs) / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h}h ${mm}m` : `${h}h`;
}

export default function TrainingRunCard({ run, onDelete, onCancel }) {
  const navigate = useNavigate();

  const isRunning = run.status === "running" || run.status === "pending";
  const finalLoss = run.metrics?.final_loss;
  const totalSteps = run.metrics?.total_steps;

  const handleOpen = () => {
    navigate(`/training/live/${run.run_id}`);
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    if (
      !confirm(
        `Eliminare definitivamente il training "${run.run_id}"? Verrà rimosso anche l'adapter dal disco.`
      )
    )
      return;
    onDelete(run.id);
  };

  const handleCancel = (e) => {
    e.stopPropagation();
    if (!confirm("Cancellare il training in corso? Lo step in esecuzione terminerà prima dello stop."))
      return;
    onCancel(run.id);
  };

  return (
    <div
      onClick={handleOpen}
      className="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-2xl p-5 cursor-pointer transition-colors group"
    >
      {/* Header riga */}
      <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <code className="text-sm text-zinc-300 font-mono">{run.run_id}</code>
            <StatusPill status={run.status} />
          </div>
          <div className="text-xs text-zinc-500 mt-1">
            #{run.id} · creato {formatDate(run.created_at)}
          </div>
        </div>
        <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleOpen}
            className="p-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors"
            title="Apri monitor"
          >
            <Eye className="w-4 h-4" />
          </button>
          {isRunning && onCancel && (
            <button
              onClick={handleCancel}
              className="p-2 rounded-lg bg-red-900/40 hover:bg-red-900/60 text-red-200 transition-colors"
              title="Cancella training"
            >
              <Ban className="w-4 h-4" />
            </button>
          )}
          {!isRunning && (
            <button
              onClick={handleDelete}
              className="p-2 rounded-lg bg-red-900/40 hover:bg-red-900/60 text-red-200 transition-colors"
              title="Elimina"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Modello + Dataset */}
      <div className="flex items-center gap-2 text-sm text-zinc-400 mb-4 flex-wrap">
        <Cpu className="w-3.5 h-3.5 text-zinc-500" />
        <span className="text-zinc-300">
          {run.base_model_name || `Modello #${run.base_model_id}`}
        </span>
        <span className="text-zinc-600">←</span>
        <Database className="w-3.5 h-3.5 text-zinc-500" />
        <span className="text-zinc-300">
          {run.dataset_name || (run.dataset_id ? `Dataset #${run.dataset_id}` : "-")}
        </span>
      </div>

      {/* Metriche */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
            <TrendingUp className="w-3 h-3" /> Loss finale
          </div>
          <div className="text-base font-semibold font-mono text-zinc-100">
            {finalLoss != null ? finalLoss.toFixed(4) : "-"}
          </div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
            <Hash className="w-3 h-3" /> Step
          </div>
          <div className="text-base font-semibold font-mono text-zinc-100">
            {totalSteps != null ? totalSteps : "-"}
          </div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
            <Clock className="w-3 h-3" /> Tempo
          </div>
          <div className="text-base font-semibold font-mono text-zinc-100">
            {formatDuration(run.started_at, run.finished_at)}
          </div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500 mb-0.5">
            Avviato
          </div>
          <div className="text-xs text-zinc-300">
            {formatDate(run.started_at)}
          </div>
        </div>
      </div>

      {/* Errore se failed */}
      {run.status === "failed" && run.error_message && (
        <div className="mt-3 text-xs text-red-300 bg-red-950/30 border border-red-900/50 rounded-lg p-2.5">
          <strong>Errore:</strong> {run.error_message}
        </div>
      )}
    </div>
  );
}