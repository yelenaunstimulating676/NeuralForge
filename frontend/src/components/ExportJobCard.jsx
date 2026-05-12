import { Loader2, X, CheckCircle2, XCircle, Ban } from "lucide-react";

export default function ExportJobCard({ job, onCancel }) {
  const pct = Math.round((job.progress || 0) * 100);

  const statusMap = {
    pending: { color: "text-zinc-400", label: "In coda", icon: Loader2 },
    running: { color: "text-indigo-300", label: "In corso", icon: Loader2, spin: true },
    completed: { color: "text-emerald-300", label: "Completato", icon: CheckCircle2 },
    failed: { color: "text-red-300", label: "Fallito", icon: XCircle },
    cancelled: { color: "text-amber-300", label: "Cancellato", icon: Ban },
  };
  const s = statusMap[job.status] || statusMap.pending;
  const Icon = s.icon;
  const isRunning = job.status === "running" || job.status === "pending";

  const filename = job.result?.output_filename || "export.gguf";

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-sm text-zinc-100 truncate">
            {filename}
          </div>
          <div className="text-[11px] text-zinc-500 mt-0.5 flex items-center gap-1.5">
            <Icon className={`w-3 h-3 ${s.spin ? "animate-spin" : ""} ${s.color}`} />
            <span className={s.color}>{s.label}</span>
            {job.result?.elapsed_seconds && (
              <>
                <span>·</span>
                <span>{job.result.elapsed_seconds.toFixed(1)}s</span>
              </>
            )}
          </div>
        </div>
        {isRunning && onCancel && (
          <button
            onClick={() => onCancel(job.job_id)}
            className="p-1.5 rounded-md hover:bg-red-900/30 text-red-300"
            title="Annulla export"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Progress bar */}
      {isRunning && (
        <>
          <div className="h-2 bg-zinc-950 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-1.5 text-[11px]">
            <span className="text-zinc-400 truncate">{job.message || "…"}</span>
            <span className="text-zinc-500 font-mono ml-2 flex-shrink-0">{pct}%</span>
          </div>
        </>
      )}

      {job.status === "failed" && job.error && (
        <div className="text-[11px] text-red-300 bg-red-950/30 border border-red-900/50 rounded p-2 mt-2 break-words">
          {job.error}
        </div>
      )}
    </div>
  );
}