import { Component } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      const errorMsg = this.state.error?.message || "Errore sconosciuto";

      return (
        <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-6">
          <div className="bg-zinc-900 border border-red-900/50 rounded-2xl p-8 max-w-lg w-full">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-8 h-8 text-red-400" />
              <h1 className="text-xl font-semibold text-zinc-100">
                Qualcosa è andato storto
              </h1>
            </div>

            <p className="text-sm text-zinc-400 mb-4">
              L'interfaccia ha incontrato un errore inatteso. Puoi provare a
              ricaricare la pagina. Se il problema persiste, controlla la
              console del browser.
            </p>

            <details className="mb-6">
              <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-300">
                Dettagli tecnici
              </summary>
              <pre className="mt-2 p-3 bg-zinc-950 border border-zinc-800 rounded text-[11px] text-red-300 overflow-x-auto whitespace-pre-wrap break-words">
                {errorMsg}
              </pre>
            </details>

            <div className="flex items-center gap-2">
              <button
                onClick={this.handleReload}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg inline-flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Ricarica pagina
              </button>
              <button
                onClick={this.handleReset}
                className="px-4 py-2 border border-zinc-700 hover:border-zinc-600 text-zinc-300 text-sm rounded-lg"
              >
                Continua
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}