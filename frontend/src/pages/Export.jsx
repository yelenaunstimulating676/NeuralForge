import { useEffect, useMemo, useState } from "react";
import {
  Package,
  Plus,
  Download,
  Trash2,
  AlertCircle,
  Sparkles,
} from "lucide-react";
import {
  fetchExportJobs,
  fetchExportFiles,
  deleteExportFile,
  cancelExportJob,
  exportFileDownloadUrl,
} from "../api/client";
import ExportModal from "../components/ExportModal";
import ExportJobCard from "../components/ExportJobCard";

function formatBytes(n) {
  if (n == null) return "â€”";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(iso) {
  if (!iso) return "â€”";
  try {
    return new Date(iso).toLocaleString("it-IT", {
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

export default function Export() {
  const [modalOpen, setModalOpen] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    try {
      const [jobsData, filesData] = await Promise.all([
        fetchExportJobs(),
        fetchExportFiles(),
      ]);
      setJobs(jobsData || []);
      setFiles(filesData || []);
      setError(null);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const hasActiveJob = useMemo(
    () => jobs.some((j) => j.status === "running" || j.status === "pending"),
    [jobs]
  );

  useEffect(() => {
    if (!hasActiveJob) return;
    const interval = setInterval(load, 2000);
    return () => clearInterval(interval);
  }, [hasActiveJob]);

  const onSubmitted = () => {
    setTimeout(load, 200);
  };

  const onCancel = async (jobId) => {
    if (!confirm("Annullare l'export in corso?")) return;
    try {
      await cancelExportJob(jobId);
      await load();
    } catch (e) {
      alert("Errore annullamento: " + (e.response?.data?.detail || e.message));
    }
  };

  const onDelete = async (filename) => {
    if (!confirm(`Eliminare il file ${filename}?`)) return;
    try {
      await deleteExportFile(filename);
      await load();
    } catch (e) {
      alert("Errore eliminazione: " + (e.response?.data?.detail || e.message));
    }
  };

  const activeJobs = jobs.filter(
    (j) => j.status === "running" || j.status === "pending"
  );
  const recentFinished = jobs
    .filter((j) => j.status !== "running" && j.status !== "pending")
    .slice(0, 3);

  if (loading) {
    return <div className="p-8 text-zinc-400">Caricamentoâ€¦</div>;
  }

  return (
    <div className="p-6 lg:p-8 space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-3xl font-bold mb-1 flex items-center gap-2">
            <Package className="w-7 h-7 text-indigo-400" />
            Export
          </h1>
          <p className="text-zinc-400">
            Esporta i tuoi modelli fine-tunati come{" "}
            <code className="text-zinc-300">.gguf</code> per usarli in Ollama,
            LM Studio, llama.cpp.
          </p>
        </div>
        <button
          onClick={() => setModalOpen(true)}
          className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Nuovo export
        </button>
      </div>

      {error && (
        <div className="bg-red-950/40 border border-red-900/60 rounded-xl p-4 text-sm text-red-200 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <strong>Errore:</strong> {error}
          </div>
        </div>
      )}

      {activeJobs.length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-2 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            Export in corso
          </h2>
          <div className="space-y-2">
            {activeJobs.map((job) => (
              <ExportJobCard
                key={job.job_id}
                job={job}
                onCancel={onCancel}
              />
            ))}
          </div>
        </section>
      )}

      {recentFinished.length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-2">
            Ultimi completati
          </h2>
          <div className="space-y-2">
            {recentFinished.map((job) => (
              <ExportJobCard key={job.job_id} job={job} />
            ))}
          </div>
        </section>
      )}

      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-zinc-300">
            File esportati{" "}
            <span className="text-zinc-500 font-normal">({files.length})</span>
          </h2>
        </div>

        {files.length === 0 ? (
          <div className="bg-zinc-900 border border-zinc-800 border-dashed rounded-2xl p-12 text-center">
            <Package className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
            <p className="text-zinc-400 mb-1">Nessun export salvato.</p>
            <p className="text-xs text-zinc-600 mb-4">
              Clicca "Nuovo export" per creare il primo .gguf.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((f) => (
              <div
                key={f.filename}
                className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex items-center gap-3 flex-wrap"
              >
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-sm text-zinc-100 truncate">
                    {f.filename}
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5 flex items-center gap-3 flex-wrap">
                    <span className="font-mono">{formatBytes(f.size_bytes)}</span>
                    <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 text-[10px] font-medium">
                      {f.quantization}
                    </span>
                    {f.ft_name && <span>da {f.ft_name}</span>}
                    <span>Creato {formatDate(f.created_at)}</span>
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <a
                    href={exportFileDownloadUrl(f.filename)}
                    download={f.filename}
                    className="px-3 py-1.5 text-xs rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Scarica
                  </a>
                  <button
                    onClick={() => onDelete(f.filename)}
                    className="px-3 py-1.5 text-xs rounded-lg bg-red-900/30 hover:bg-red-900/50 text-red-200 flex items-center gap-1.5"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Elimina
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <ExportModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmitted={onSubmitted}
      />
    </div>
  );
}