import { useNavigate } from "react-router-dom";
import {
  Boxes,
  Database,
  Cpu,
  CheckCircle2,
  ArrowRight,
  Lock,
  Zap,
} from "lucide-react";

/**
 * Quick Start Guide — sezione di onboarding mostrata in Dashboard
 * quando l'utente non ha ancora completato i 3 step base.
 *
 * Stati delle card:
 *   - done    : step completato (verde)
 *   - active  : step da fare ora (indigo, cliccabile)
 *   - locked  : step bloccato (grigio, prerequisito mancante)
 */
export default function QuickStartGuide({ hasModels, hasDatasets, hasTrainings }) {
  const navigate = useNavigate();

  // Se tutti e 3 i passi sono fatti, non mostriamo nulla
  if (hasModels && hasDatasets && hasTrainings) return null;

  const steps = [
    {
      id: 1,
      label: "Scarica un modello",
      description: "Vai sulla pagina Models e scarica un base model da HuggingFace.",
      icon: Boxes,
      done: hasModels,
      active: !hasModels,
      route: "/models",
    },
    {
      id: 2,
      label: "Carica un dataset",
      description: "Crea un dataset in formato alpaca (JSONL) per il fine-tuning.",
      icon: Database,
      done: hasDatasets,
      active: hasModels && !hasDatasets,
      locked: !hasModels,
      route: "/dataset",
    },
    {
      id: 3,
      label: "Avvia un training",
      description: "Fine-tuna il tuo primo modello con QLoRA.",
      icon: Cpu,
      done: hasTrainings,
      active: hasModels && hasDatasets && !hasTrainings,
      locked: !(hasModels && hasDatasets),
      route: "/training",
    },
  ];

  return (
    <div className="rounded-2xl border border-indigo-900/40 bg-gradient-to-br from-indigo-950/30 to-zinc-950 p-6">
      <div className="flex items-center gap-2 mb-1">
        <Zap className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-zinc-100">
          Benvenuto in NeuralForge
        </h2>
      </div>
      <p className="text-sm text-zinc-400 mb-5">
        Tre passi per il tuo primo fine-tuning locale.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {steps.map((step, idx) => {
          const StepIcon = step.icon;
          const isLast = idx === steps.length - 1;

          // Stato visivo
          let cardClass, iconClass, badge;
          if (step.done) {
            cardClass = "border-emerald-700/50 bg-emerald-950/20";
            iconClass = "text-emerald-400";
            badge = (
              <div className="flex items-center gap-1 text-xs text-emerald-300">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Completato
              </div>
            );
          } else if (step.locked) {
            cardClass = "border-zinc-800 bg-zinc-950/50 opacity-50";
            iconClass = "text-zinc-600";
            badge = (
              <div className="flex items-center gap-1 text-xs text-zinc-500">
                <Lock className="w-3.5 h-3.5" />
                Bloccato
              </div>
            );
          } else if (step.active) {
            cardClass =
              "border-indigo-600/60 bg-indigo-950/30 hover:bg-indigo-950/50 cursor-pointer ring-2 ring-indigo-600/30";
            iconClass = "text-indigo-400";
            badge = (
              <div className="flex items-center gap-1 text-xs text-indigo-300 font-medium">
                Inizia
                <ArrowRight className="w-3.5 h-3.5" />
              </div>
            );
          } else {
            cardClass = "border-zinc-800 bg-zinc-900";
            iconClass = "text-zinc-500";
            badge = null;
          }

          return (
            <div key={step.id} className="relative">
              <button
                onClick={() => !step.locked && navigate(step.route)}
                disabled={step.locked}
                className={`w-full text-left p-4 rounded-xl border transition-colors ${cardClass}`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-zinc-500">
                      {step.id}.
                    </span>
                    <StepIcon className={`w-5 h-5 ${iconClass}`} />
                  </div>
                  {badge}
                </div>
                <div className="font-medium text-zinc-100 text-sm mb-1">
                  {step.label}
                </div>
                <div className="text-xs text-zinc-500 leading-relaxed">
                  {step.description}
                </div>
              </button>

              {/* Connettore visivo tra step (solo desktop) */}
              {!isLast && (
                <div className="hidden md:block absolute top-1/2 -right-2 transform -translate-y-1/2 z-10">
                  <ArrowRight className="w-4 h-4 text-zinc-700" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}