import { useState } from "react";
import { Settings2, ChevronDown } from "lucide-react";

const PRESETS = {
  preciso: {
    label: "Preciso",
    description: "T=0.1 · risposte deterministiche",
    params: { temperature: 0.1, top_p: 0.5, top_k: 30, repetition_penalty: 1.2 },
  },
  bilanciato: {
    label: "Bilanciato",
    description: "Default · buon compromesso",
    params: { temperature: 0.7, top_p: 0.9, top_k: 50, repetition_penalty: 1.1 },
  },
  creativo: {
    label: "Creativo",
    description: "T=1.0 · risposte varie",
    params: { temperature: 1.0, top_p: 0.95, top_k: 100, repetition_penalty: 1.05 },
  },
};

function Slider({ label, value, min, max, step, onChange, format }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label className="text-[11px] uppercase tracking-wider text-zinc-500">
          {label}
        </label>
        <span className="text-xs font-mono text-zinc-300">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

export default function SamplingControls({ params, onChange }) {
  const [preset, setPreset] = useState("bilanciato");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const applyPreset = (key) => {
    setPreset(key);
    onChange({ ...params, ...PRESETS[key].params });
  };

  const updateParam = (k, v) => {
    setPreset("custom");
    onChange({ ...params, [k]: v });
  };

  return (
    <div>
      {/* Preset buttons */}
      <div className="flex items-center gap-2 flex-wrap">
        {Object.entries(PRESETS).map(([key, p]) => (
          <button
            key={key}
            onClick={() => applyPreset(key)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              preset === key
                ? "border-indigo-500 bg-indigo-500/10 text-indigo-200"
                : "border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-zinc-300"
            }`}
            title={p.description}
          >
            {p.label}
          </button>
        ))}
        {preset === "custom" && (
          <span className="px-3 py-1.5 text-xs rounded-lg border border-amber-700/50 bg-amber-900/20 text-amber-300">
            Personalizzato
          </span>
        )}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="ml-auto px-2.5 py-1.5 text-xs rounded-lg border border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-zinc-400 flex items-center gap-1.5"
        >
          <Settings2 className="w-3.5 h-3.5" />
          Avanzate
          <ChevronDown
            className={`w-3 h-3 transition-transform ${showAdvanced ? "rotate-180" : ""}`}
          />
        </button>
      </div>

      {/* Advanced sliders */}
      {showAdvanced && (
        <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 p-4 bg-zinc-950 border border-zinc-800 rounded-lg">
          <Slider
            label="Temperature"
            value={params.temperature}
            min={0.05}
            max={2.0}
            step={0.05}
            onChange={(v) => updateParam("temperature", v)}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Top-p"
            value={params.top_p}
            min={0.05}
            max={1.0}
            step={0.05}
            onChange={(v) => updateParam("top_p", v)}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Top-k"
            value={params.top_k}
            min={0}
            max={200}
            step={1}
            onChange={(v) => updateParam("top_k", v)}
          />
          <Slider
            label="Repetition penalty"
            value={params.repetition_penalty}
            min={1.0}
            max={2.0}
            step={0.05}
            onChange={(v) => updateParam("repetition_penalty", v)}
            format={(v) => v.toFixed(2)}
          />
          <div className="col-span-2">
            <Slider
              label="Max new tokens"
              value={params.max_new_tokens}
              min={16}
              max={1024}
              step={16}
              onChange={(v) => updateParam("max_new_tokens", v)}
            />
          </div>
        </div>
      )}
    </div>
  );
}