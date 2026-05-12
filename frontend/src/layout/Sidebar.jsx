/**
 * Sidebar di navigazione principale.
 * Le 5 voci corrispondono alle pagine della roadmap.
 */

import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Database,
  Cpu,
  Activity,
  Boxes,
  MessageSquare,
  Package,
  Zap,
} from 'lucide-react'

const links = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/training', label: 'Training', icon: Cpu },
  { to: '/inference', label: 'Inference', icon: MessageSquare },
  { to: '/monitor', label: 'Monitor', icon: Activity },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/export', label: 'Export', icon: Package },
]

export default function Sidebar() {
  return (
    <aside className="flex h-screen w-60 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)]">
      {/* Logo / Brand */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-[var(--color-border)]">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-[var(--color-accent)]">
          <Zap size={18} className="text-white" />
        </div>
        <div className="flex flex-col leading-tight">
          <span className="font-semibold text-[var(--color-text)]">
            NeuralForge
          </span>
          <span className="text-xs text-[var(--color-text-muted)]">
            v0.1.0
          </span>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              [
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-[var(--color-surface-2)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text)]',
              ].join(' ')
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-[var(--color-border)] text-xs text-[var(--color-text-muted)]">
        Local LLM fine-tuning
      </div>
    </aside>
  )
}