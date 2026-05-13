import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, RefreshCw, Sparkles } from "lucide-react";
import {
  fetchTrainingRuns,
  cancelTrainingRun,
  deleteTrainingRun,
} from "../api/client";
import TrainingRunCard from "../components/TrainingRunCard";
import useDocumentTitle from "../hooks/useDocumentTitle";
import PageLoader from "../components/PageLoader";

const FILTERS = [
  { value: "all", label: "Tutti" },
  { value: "running", label: "In corso" },
  { value: "completed", label: "Completati" },
  { value: "failed", label: "Falliti" },
  { value: "cancelled", label: "Cancellati" },
];

export default function Monitor() {
  useDocumentTitle("Monitor");
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all");
  const [refreshing, setRefreshing] = useState(false);

  const loadRuns = async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    try {
      const data = await fetchTrainingRuns();
      setRuns(data || []);
      setError(null);
    } catch (e) {
      console.error("Errore caricamento runs:", e);
      setError(e.response?.data?.detail || e.message || "Errore");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, []);

  // Auto-refresh ogni 3s SOLO se c'è almeno un run in corso
  const hasRunning = runs.some(
    (r) => r.status === "running" || r.status === "pending"
  );

  useEffect(() => {
    if (!hasRunning) return;
    const interval = setInterval(() => {
      loadRuns(false);
    }, 3000);
    return () => clearInterval(interval);
  }, [hasRunning]);

  // Statistiche aggregate
  const stats = useMemo(() => {
    const counts = {
      total: runs.length,
      running: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      pending: 0,
    };
    for (const r of runs) {
      counts[r.status] = (counts[r.status] || 0) + 1;
    }
    return counts;
  }, [runs]);

  // Filtered list
  const filtered = useMemo(() => {
    if (filter === "all") return runs;
    if (filter === "running") {
      return runs.filter((r) => r.status === "running" || r.status === "pending");
    }
    return runs.filter((r) => r.status === filter);
  }, [runs, filter]);

  const onCancel = async (id) => {
    try {
      await cancelTrainingRun(id);
      await loadRuns();
    } catch (e) {
      alert("Errore cancellazione: " + (e.response?.data?.detail || e.message));
    }
  };

  const onDelete = async (id) => {
    try {
      await deleteTrainingRun(id);
      setRuns((prev) => prev.filter((r) => r.id !== id));
    } catch (e) {
      alert("Errore eliminazione: " + (e.response?.data?.detail || e.message));
    }
  };

  if (loading) {
    return <PageLoader message="Caricamento storico training..." />;
  }

  if (error) {
    return (
      <div className="p-8 max-w-2xl">
        <h1 className="text-3xl font-bold mb-2">Monitor</h1>
        <div className="bg-red-950/40 border border-red-900/60 rounded-xl p-4 text-sm text-red-200">
          <strong>Errore:</strong> {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-6 flex-wrap">
        <div>
          <h1 className="text-3xl font-bold mb-1 flex items-center gap-2">
            <Activity className="w-7 h-7 text-indigo-400" />
            Monitor
          </h1>
          <p className="text-zinc-400">
            Storico di tutti i training run. Click su una card per il monitor live.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => loadRuns(true)}
            disabled={refreshing}
            className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 text-sm rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
            Aggiorna
          </button>
          <button
            onClick={() => navigate("/training")}
            className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg flex items-center gap-2 transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            Nuovo training
          </button>
        </div>
      </div>

      {/* Stats summary */}
      {runs.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 mb-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <span className="text-zinc-400">
            Totale: <strong className="text-zinc-100">{stats.total}</strong>
          </span>
          {stats.running > 0 && (
            <span className="text-indigo-300">
              ● {stats.running} in corso
            </span>
          )}
          {stats.completed > 0 && (
            <span className="text-emerald-300">
              ✓ {stats.completed} completati
            </span>
          )}
          {stats.cancelled > 0 && (
            <span className="text-amber-300">
              ⊘ {stats.cancelled} cancellati
            </span>
          )}
          {stats.failed > 0 && (
            <span className="text-red-300">
              ✕ {stats.failed} falliti
            </span>
          )}
          {hasRunning && (
            <span className="ml-auto text-xs text-zinc-500 italic">
              Auto-refresh attivo (3s)
            </span>
          )}
        </div>
      )}

      {/* Empty state */}
      {runs.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 border-dashed rounded-2xl p-12 text-center">
          <Activity className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
          <p className="text-zinc-400 mb-1">Nessun training avviato.</p>
          <p className="text-xs text-zinc-600 mb-4">
            Quando avvierai un fine-tuning, lo vedrai qui con metriche live.
          </p>
          <button
            onClick={() => navigate("/training")}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg inline-flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            Avvia il primo training
          </button>
        </div>
      ) : (
        <>
          {/* Filtri */}
          <div className="flex gap-1.5 mb-4 flex-wrap">
            {FILTERS.map((f) => {
              const count =
                f.value === "all"
                  ? stats.total
                  : f.value === "running"
                  ? stats.running + stats.pending
                  : stats[f.value] || 0;
              const active = filter === f.value;
              return (
                <button
                  key={f.value}
                  onClick={() => setFilter(f.value)}
                  className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
                    active
                      ? "border-indigo-500 bg-indigo-500/10 text-indigo-200"
                      : "border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-zinc-400"
                  }`}
                >
                  {f.label}{" "}
                  <span className={active ? "text-indigo-400" : "text-zinc-600"}>
                    ({count})
                  </span>
                </button>
              );
            })}
          </div>

          {/* Lista cards */}
          {filtered.length === 0 ? (
            <div className="text-zinc-500 text-sm py-8 text-center">
              Nessun run con stato "{filter}".
            </div>
          ) : (
            <div className="space-y-3">
              {filtered.map((run) => (
                <TrainingRunCard
                  key={run.id}
                  run={run}
                  onCancel={onCancel}
                  onDelete={onDelete}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}