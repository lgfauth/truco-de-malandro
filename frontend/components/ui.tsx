"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const NAV = [
  { href: "/play", label: "Jogar" },
  { href: "/watch", label: "Treino" },
  { href: "/runs", label: "Runs" },
  { href: "/arena", label: "Arena" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
      <Link href="/" className="font-bold text-zinc-100 mr-2">
        Truco RL
      </Link>
      {NAV.map((n) => {
        const active = path === n.href || path.startsWith(`${n.href}/`);
        return (
          <Link
            key={n.href}
            href={n.href}
            className={
              active
                ? "text-emerald-400 font-semibold"
                : "text-zinc-400 hover:text-zinc-100"
            }
          >
            {n.label}
          </Link>
        );
      })}
    </nav>
  );
}

export function Page({
  title,
  subtitle,
  actions,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main className="min-h-screen px-4 sm:px-6 py-6 sm:py-8 max-w-6xl mx-auto space-y-6">
      <Nav />
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl sm:text-3xl font-bold break-words">{title}</h1>
          {subtitle && <div className="text-sm text-zinc-400 mt-1">{subtitle}</div>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </header>
      {children}
    </main>
  );
}

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-zinc-800 bg-zinc-900 p-4 min-w-0 ${className}`}>
      {(title || right) && (
        <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>}
            {subtitle && <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4 min-w-0">
      <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-1 text-2xl font-bold text-zinc-100 truncate">{value}</div>
      {hint && <div className="text-xs text-zinc-500 mt-1">{hint}</div>}
    </div>
  );
}

export function Loading({ label = "Carregando…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-zinc-400 py-6" role="status">
      <span className="h-4 w-4 rounded-full border-2 border-zinc-600 border-t-emerald-400 animate-spin" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="p-3 rounded-lg bg-rose-950/70 border border-rose-800 text-rose-200 text-sm flex flex-wrap items-center justify-between gap-2"
    >
      <span className="break-all">⚠ {error}</span>
      {onRetry && (
        <button onClick={onRetry} className="px-3 py-1 rounded bg-rose-800 hover:bg-rose-700 text-xs font-semibold">
          Tentar de novo
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-zinc-700 p-6 text-center text-sm text-zinc-400">
      {children}
    </div>
  );
}

/** Status callout: always icon + label, never color alone. */
export function Warning({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div role="status" className="rounded-lg border border-[#fab219]/60 bg-[#fab219]/10 p-3 text-sm">
      <div className="font-semibold text-zinc-100">⚠ {title}</div>
      <div className="text-zinc-300 mt-1">{children}</div>
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    running: ["● rodando", "text-emerald-300 border-emerald-700"],
    finished: ["✓ concluída", "text-zinc-200 border-zinc-600"],
    stopped: ["■ parada", "text-zinc-300 border-zinc-600"],
    failed: ["✕ falhou", "text-rose-300 border-rose-800"],
    queued: ["… na fila", "text-zinc-300 border-zinc-600"],
    done: ["✓ pronto", "text-zinc-200 border-zinc-600"],
  };
  const [label, cls] = map[status] ?? [status, "text-zinc-300 border-zinc-600"];
  return (
    <span className={`inline-block whitespace-nowrap px-2 py-0.5 rounded border text-xs ${cls}`}>
      {label}
    </span>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  tone = "zinc",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "zinc" | "emerald" | "amber" | "rose";
  type?: "button" | "submit";
}) {
  const tones = {
    zinc: "bg-zinc-700 hover:bg-zinc-600 text-zinc-100",
    emerald: "bg-emerald-600 hover:bg-emerald-500 text-white",
    amber: "bg-amber-500 hover:bg-amber-400 text-zinc-900",
    rose: "bg-rose-700 hover:bg-rose-600 text-white",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 rounded-lg font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed ${tones[tone]}`}
    >
      {children}
    </button>
  );
}
