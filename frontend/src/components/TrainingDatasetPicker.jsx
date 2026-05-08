export default function TrainingDatasetPicker({ datasets, selectedId, onSelect }) {
  if (!datasets.length) {
    return <p className="text-sm text-zinc-500">Nessun dataset disponibile.</p>;
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {datasets.map((d) => {
        const selected = selectedId === d.id;
        return (
          <button
            key={d.id}
            onClick={() => onSelect(d.id)}
            className={`text-left p-3 rounded-xl border transition-colors ${
              selected
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-zinc-800 hover:border-zinc-700 bg-zinc-950"
            }`}
          >
            <div className="flex items-start gap-2">
              <div className={`mt-1 w-3.5 h-3.5 rounded-full border-2 flex-shrink-0 ${
                selected ? "border-indigo-400 bg-indigo-500" : "border-zinc-600"
              }`} />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{d.name}</div>
                <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-1.5 text-[11px] text-zinc-400">
                  <span>{d.num_examples} esempi</span>
                  <span className="px-1 py-0.5 rounded bg-zinc-800 font-mono text-[10px]">
                    {d.format}
                  </span>
                </div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}