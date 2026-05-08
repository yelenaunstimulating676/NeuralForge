import { useEffect, useRef, useState, useMemo } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  Cpu,
  Clock,
  Hash,
  TrendingUp,
  Zap,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Loader,
  ArrowLeft,
  Ban,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  trainingWebSocketUrl,
  fetchTrainingRuns,
  cancelTrainingRun,
} from "../api/client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(seconds) {
  if (!seconds || seconds < 1) return "<1s";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h}h ${mm}m` : `${h}h`;
}

function formatNumberShort(n) {
  if (n == null) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  if (Math.abs(n) >= 0.001) return n.toFixed(4);
  return n.toExponential(2);
}

function StatusBadge({ status }) {
  const map = {
    pending: { color: "bg-zinc-700 text-zinc-300", label: "In coda", icon: Loader },
    running: { color: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40", label: "In corso", icon: Loader, spin: true },
    completed: { color: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40", label: "Completato", icon: CheckCircle2 },
    failed: { color: "bg-red-500/20 text-red-300 border-red-500/40", label: "Fallito", icon: XCircle },
    cancelled: { color: "bg-amber-500/20 text-amber-300 border-amber-500/40", label: "Cancellato", icon: Ban },
  };
  const conf = map[status] || map.pending;
  const Icon = conf.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-medium ${conf.color}`}>
      <Icon className={`w-3.5 h-3.5 ${conf.spin ? "animate-spin" : ""}`} />
      {conf.label}
    </span>
  );
}

function StatBox({ icon: Icon, label, value, sublabel }) {
  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
        <Icon className="w-3 h-3" />
        {label}
      </div>
      <div className="text-xl font-semibold text-zinc-100 font-mono">{value}</div>
      {sublabel && <div className="text-[10px] text-zinc-500 mt-0.5">{sublabel}</div>}
    </div>
  );
}

function MetricChart({ data, dataKey, color, label, yFormatter }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-zinc-600 text-sm">
        In attesa di dati…
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis
          dataKey="step"
          stroke="#71717a"
          tick={{ fontSize: 10 }}
          label={{ value: "step", position: "insideBottom", fill: "#52525b", fontSize: 10, dy: 10 }}
        />
        <YAxis
          stroke="#71717a"
          tick={{ fontSize: 10 }}
          tickFormatter={yFormatter}
          width={50}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#09090b",
            border: "1px solid #3f3f46",
            borderRadius: "8px",
            fontSize: "12px",
          }}
          labelStyle={{ color: "#a1a1aa" }}
          formatter={(v) => [yFormatter ? yFormatter(v) : v, label]}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function TrainingLive() {
  const { runId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const initialJobId = location.state?.jobId;

  const [status, setStatus] = useState("pending");
  const [stepLogs, setStepLogs] = useState([]); // array di {step, loss, lr, ...}
  const [finishedEvent, setFinishedEvent] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  const [dbId, setDbId] = useState(null); // training_run_db_id (per cancel)
  const [cancelling, setCancelling] = useState(false);

  const wsRef = useRef(null);
  const logsContainerRef = useRef(null);

// Carica db_id e metadata cercando per run_id (ora persistito in DB)
  const [runMeta, setRunMeta] = useState(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const runs = await fetchTrainingRuns();
        if (cancelled) return;
        const match = (runs || []).find((r) => r.run_id === runId);
        if (match) {
          setDbId(match.id);
          setRunMeta(match);
          // Se il run è già completato in DB, popola anche stato + history
          if (match.status !== "running" && match.status !== "pending") {
            setStatus(match.status);
            if (match.metrics) {
              setStepLogs(match.metrics.history || []);
              if (!finishedEvent) {
                setFinishedEvent({
                  status: match.status,
                  final_loss: match.metrics.final_loss,
                  total_steps: match.metrics.total_steps,
                  elapsed_seconds: match.metrics.elapsed_seconds,
                  error: match.error_message,
                });
              }
            }
          }
        }
      } catch (e) {
        console.warn("Impossibile recuperare db_id:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [runId]);

  // WebSocket connection
  useEffect(() => {
    const wsUrl = trainingWebSocketUrl(runId);
    console.log("[WS] connecting to", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[WS] connected");
      setWsConnected(true);
    };

    ws.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data);
        if (evt.type === "step_log") {
          setStepLogs((prev) => [...prev, evt.data]);
        } else if (evt.type === "status_change") {
          setStatus(evt.data.status || "running");
        } else if (evt.type === "finished") {
          setFinishedEvent(evt.data);
          setStatus(evt.data.status || "completed");
        } else if (evt.type === "error") {
          setErrorMsg(evt.data.message || "Errore sconosciuto");
          setStatus("failed");
        }
      } catch (err) {
        console.error("[WS] parse error:", err);
      }
    };

    ws.onclose = (e) => {
      console.log("[WS] closed", e.code);
      setWsConnected(false);
    };

    ws.onerror = (e) => {
      console.error("[WS] error:", e);
      setWsConnected(false);
    };

    return () => {
      try {
        ws.close();
      } catch {}
    };
  }, [runId]);

  // Auto-scroll dei log
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [stepLogs.length]);

  // Stats correnti (ultimo step)
  const lastStep = stepLogs[stepLogs.length - 1];

  // ETA stima
  const eta = useMemo(() => {
    if (!lastStep || stepLogs.length < 3) return null;
    const elapsed = lastStep.elapsed_seconds;
    const rate = lastStep.step / elapsed; // step/s
    // Non sappiamo total_steps senza chiedere al backend, tentiamo di stimarlo
    // dal trend: assumiamo che il training continui per X step in più.
    // In assenza di total_steps, mostriamo solo elapsed. ETA solo se finishedEvent ha total.
    return null;
  }, [lastStep, stepLogs.length]);

  // Total steps dal finishedEvent (quando il run è già terminato)
  const totalSteps = finishedEvent?.total_steps;
  const progress = totalSteps && lastStep ? lastStep.step / totalSteps : null;

  const isRunning = status === "running" || status === "pending";

  const onCancel = async () => {
    if (!dbId || cancelling) return;
    if (!confirm("Vuoi davvero cancellare il training? Lo step in corso terminerà prima dello stop.")) {
      return;
    }
    setCancelling(true);
    try {
      await cancelTrainingRun(dbId);
    } catch (e) {
      console.error("Cancel error:", e);
      alert("Errore cancellazione: " + (e.response?.data?.detail || e.message));
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6 flex items-start justify-between gap-4 flex-wrap">
        <div>
          <button
            onClick={() => navigate("/training")}
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 mb-2 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Torna a Training
          </button>
          <h1 className="text-3xl font-bold mb-1 flex items-center gap-2">
            <Activity className="w-7 h-7 text-indigo-400" />
            Live Monitor
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <code className="text-sm text-zinc-400 font-mono">{runId}</code>
            <StatusBadge status={status} />
            {!wsConnected && isRunning && (
              <span className="text-xs text-amber-400">WS disconnesso</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && dbId && (
            <button
              disabled={cancelling}
              onClick={onCancel}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:bg-zinc-800 text-white text-sm rounded-lg flex items-center gap-2"
            >
              <Ban className="w-4 h-4" />
              {cancelling ? "Cancellazione…" : "Cancella"}
            </button>
          )}
        </div>
      </div>

      {/* Banner finished/error */}
      {finishedEvent && (
        <div
          className={`mb-6 rounded-2xl p-5 border ${
            finishedEvent.status === "completed"
              ? "bg-emerald-950/30 border-emerald-800/50"
              : finishedEvent.status === "cancelled"
              ? "bg-amber-950/30 border-amber-800/50"
              : "bg-red-950/30 border-red-800/50"
          }`}
        >
          <div className="flex items-start gap-3">
            {finishedEvent.status === "completed" ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0" />
            ) : finishedEvent.status === "cancelled" ? (
              <Ban className="w-6 h-6 text-amber-400 flex-shrink-0" />
            ) : (
              <XCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
            )}
            <div className="flex-1">
              <div className="font-semibold text-zinc-100 mb-1">
                Training {finishedEvent.status}
              </div>
              <div className="text-sm text-zinc-400 grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-1">
                <div>
                  Loss finale:{" "}
                  <span className="font-mono text-zinc-200">
                    {finishedEvent.final_loss?.toFixed(4)}
                  </span>
                </div>
                <div>
                  Step totali:{" "}
                  <span className="font-mono text-zinc-200">
                    {finishedEvent.total_steps}
                  </span>
                </div>
                <div>
                  Tempo:{" "}
                  <span className="font-mono text-zinc-200">
                    {formatTime(finishedEvent.elapsed_seconds)}
                  </span>
                </div>
                {finishedEvent.finetuned_model_id && (
                  <div>
                    FT model id:{" "}
                    <span className="font-mono text-zinc-200">
                      #{finishedEvent.finetuned_model_id}
                    </span>
                  </div>
                )}
              </div>
              {finishedEvent.error && (
                <div className="mt-2 text-sm text-red-300">
                  Errore: {finishedEvent.error}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {errorMsg && !finishedEvent && (
        <div className="mb-6 bg-red-950/40 border border-red-900/60 rounded-xl p-4 text-sm text-red-200 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <strong>Errore:</strong> {errorMsg}
          </div>
        </div>
      )}

      {/* Stats correnti */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        <StatBox
          icon={Hash}
          label="Step"
          value={lastStep ? lastStep.step : "—"}
          sublabel={totalSteps ? `di ${totalSteps}` : null}
        />
        <StatBox
          icon={TrendingUp}
          label="Loss"
          value={lastStep ? lastStep.loss.toFixed(4) : "—"}
        />
        <StatBox
          icon={Zap}
          label="Learning rate"
          value={lastStep ? formatNumberShort(lastStep.learning_rate) : "—"}
        />
        <StatBox
          icon={Cpu}
          label="VRAM"
          value={lastStep ? `${Math.round(lastStep.vram_used_mb)} MB` : "—"}
        />
        <StatBox
          icon={Activity}
          label="Throughput"
          value={lastStep ? `${Math.round(lastStep.throughput_tokens_per_sec)} tok/s` : "—"}
        />
        <StatBox
          icon={Clock}
          label="Elapsed"
          value={lastStep ? formatTime(lastStep.elapsed_seconds) : "—"}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {/* Loss chart */}
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-zinc-100 mb-3">
            Loss
          </h3>
          <MetricChart
            data={stepLogs}
            dataKey="loss"
            color="#818cf8"
            label="loss"
            yFormatter={(v) => v.toFixed(2)}
          />
        </section>

        {/* LR chart */}
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-zinc-100 mb-3">
            Learning rate
          </h3>
          <MetricChart
            data={stepLogs}
            dataKey="learning_rate"
            color="#34d399"
            label="lr"
            yFormatter={(v) => v.toExponential(1)}
          />
        </section>

        {/* VRAM */}
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-zinc-100 mb-3">VRAM</h3>
          <MetricChart
            data={stepLogs}
            dataKey="vram_used_mb"
            color="#fbbf24"
            label="MB"
            yFormatter={(v) => `${Math.round(v)}`}
          />
        </section>

        {/* Throughput */}
        <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-zinc-100 mb-3">
            Throughput (tok/s)
          </h3>
          <MetricChart
            data={stepLogs}
            dataKey="throughput_tokens_per_sec"
            color="#f472b6"
            label="tok/s"
            yFormatter={(v) => Math.round(v).toString()}
          />
        </section>
      </div>

      {/* Log stream */}
      <section className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-zinc-100">Log stream</h3>
          <span className="text-xs text-zinc-500">
            {stepLogs.length} eventi
          </span>
        </div>
        <div
          ref={logsContainerRef}
          className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 h-72 overflow-y-auto font-mono text-[11px] space-y-0.5"
        >
          {stepLogs.length === 0 ? (
            <div className="text-zinc-600 text-center pt-20">
              In attesa del primo evento di training…
            </div>
          ) : (
            stepLogs.map((s, i) => (
              <div key={i} className="text-zinc-400 hover:bg-zinc-900 px-1 rounded">
                <span className="text-zinc-600">step {String(s.step).padStart(3)}</span>
                <span className="text-zinc-300 mx-2">epoch {s.epoch}</span>
                <span className="text-indigo-300">loss={s.loss.toFixed(4)}</span>
                <span className="text-emerald-300 mx-2">lr={s.learning_rate.toExponential(2)}</span>
                <span className="text-amber-300">grad={s.grad_norm.toFixed(2)}</span>
                <span className="text-pink-300 mx-2">{Math.round(s.vram_used_mb)}MB</span>
                <span className="text-zinc-500">{Math.round(s.throughput_tokens_per_sec)} tok/s</span>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}