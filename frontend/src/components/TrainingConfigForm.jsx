import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

function NumSlider({ label, value, min, max, step = 1, onChange, hint }) {
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <label className="text-xs text-zinc-400">{label}</label>
        <span className="text-sm font-mono text-zinc-200">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-indigo-500"
      />
      {hint && <p className="text-[10px] text-zinc-600 mt-0.5">{hint}</p>}
    </div>
  );
}

function NumInput({ label, value, onChange, step = 1, min, max, hint }) {
  return (
    <div>
      <label className="block text-xs text-zinc-400 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-sm font-mono"
      />
      {hint && <p className="text-[10px] text-zinc-600 mt-0.5">{hint}</p>}
    </div>
  );
}

const LR_PRESETS = [
  { label: "1e-4", value: 1e-4, hint: "cauto" },
  { label: "2e-4", value: 2e-4, hint: "default" },
  { label: "5e-4", value: 5e-4, hint: "aggressivo" },
  { label: "1e-3", value: 1e-3, hint: "molto aggr." },
];

const RANK_PRESETS = [4, 8, 16, 32, 64];

export default function TrainingConfigForm({ config, onChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const update = (patch) => onChange({ ...config, ...patch });

  return (
    <div className="space-y-5">
      {/* Sliders + Presets in due colonne */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
        {/* Colonna 1: sliders numerici */}
        <div className="space-y-4">
          <NumSlider
            label="Epochs"
            value={config.num_epochs}
            min={1}
            max={50}
            onChange={(v) => update({ num_epochs: v })}
            hint="Quante volte il training passa sull'intero dataset"
          />

          <NumSlider
            label="Batch size per device"
            value={config.per_device_batch_size}
            min={1}
            max={8}
            onChange={(v) => update({ per_device_batch_size: v })}
            hint="Esempi per micro-batch sul GPU"
          />

          <NumSlider
            label="Gradient accumulation"
            value={config.grad_accum_steps}
            min={1}
            max={16}
            onChange={(v) => update({ grad_accum_steps: v })}
            hint={`Batch effettivo: ${
              config.per_device_batch_size * config.grad_accum_steps
            }`}
          />

          <NumSlider
            label="Max sequence length"
            value={config.max_seq_length}
            min={128}
            max={4096}
            step={128}
            onChange={(v) => update({ max_seq_length: v })}
            hint="Lunghezza massima sequenza in token"
          />
        </div>

        {/* Colonna 2: preset */}
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              Learning rate
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {LR_PRESETS.map((p) => {
                const sel = Math.abs(config.learning_rate - p.value) < 1e-9;
                return (
                  <button
                    key={p.value}
                    onClick={() => update({ learning_rate: p.value })}
                    className={`text-left text-xs px-3 py-2 rounded-lg border transition-colors ${
                      sel
                        ? "border-indigo-500 bg-indigo-500/10 text-indigo-200"
                        : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
                    }`}
                  >
                    <div className="font-mono">{p.label}</div>
                    <div className="text-[10px] text-zinc-500">{p.hint}</div>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              LoRA rank
            </label>
            <div className="grid grid-cols-5 gap-1.5">
              {RANK_PRESETS.map((r) => (
                <button
                  key={r}
                  onClick={() => update({ lora_r: r, lora_alpha: r * 2 })}
                  className={`text-xs px-2 py-2 rounded-lg border transition-colors font-mono ${
                    config.lora_r === r
                      ? "border-indigo-500 bg-indigo-500/10 text-indigo-200"
                      : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-zinc-600 mt-1">
              alpha = rank × 2 = {config.lora_alpha}
            </p>
          </div>

          <div className="space-y-2 pt-1">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_4bit}
                onChange={(e) => update({ use_4bit: e.target.checked })}
                className="accent-indigo-500"
              />
              <span className="text-xs text-zinc-300">
                Quantizzazione 4-bit
                <span className="text-zinc-500 ml-1">(consigliato)</span>
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_8bit_optimizer}
                onChange={(e) => update({ use_8bit_optimizer: e.target.checked })}
                className="accent-indigo-500"
              />
              <span className="text-xs text-zinc-300">
                AdamW 8-bit
                <span className="text-zinc-500 ml-1">(consigliato)</span>
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Avanzate */}
      <div className="pt-2 border-t border-zinc-800">
        <button
          onClick={() => setShowAdvanced((s) => !s)}
          className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          {showAdvanced ? (
            <ChevronDown className="w-3.5 h-3.5" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5" />
          )}
          Opzioni avanzate
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 mt-4">
            <NumInput
              label="LoRA dropout"
              value={config.lora_dropout}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(v) => update({ lora_dropout: v })}
            />
            <NumInput
              label="Weight decay"
              value={config.weight_decay}
              min={0}
              max={0.5}
              step={0.001}
              onChange={(v) => update({ weight_decay: v })}
            />
            <NumInput
              label="Warmup ratio"
              value={config.warmup_ratio}
              min={0}
              max={0.3}
              step={0.01}
              onChange={(v) => update({ warmup_ratio: v })}
              hint="Frazione step di warmup"
            />
            <NumInput
              label="Min LR ratio"
              value={config.min_lr_ratio}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => update({ min_lr_ratio: v })}
              hint="LR finale = LR × ratio"
            />
            <NumInput
              label="Log every N steps"
              value={config.log_every_n_steps}
              min={1}
              max={100}
              step={1}
              onChange={(v) => update({ log_every_n_steps: v })}
              hint="1 = log a ogni step (per live monitor)"
            />
            <NumInput
              label="Save every N steps"
              value={config.save_every_n_steps}
              min={0}
              max={1000}
              step={10}
              onChange={(v) => update({ save_every_n_steps: v })}
              hint="0 = solo final"
            />
            <NumInput
              label="Max steps"
              value={config.max_steps}
              min={0}
              max={100000}
              step={50}
              onChange={(v) => update({ max_steps: v })}
              hint="0 = no limite"
            />
          </div>
        )}
      </div>
    </div>
  );
}