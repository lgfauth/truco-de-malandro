import Link from "next/link";

const LINKS = [
  { href: "/play", label: "Jogar contra a IA", tone: "bg-amber-600 hover:bg-amber-500" },
  { href: "/runs", label: "Runs de treino", tone: "bg-emerald-600 hover:bg-emerald-500" },
  { href: "/arena", label: "Arena", tone: "bg-sky-700 hover:bg-sky-600" },
  { href: "/watch", label: "Treinar agora", tone: "bg-zinc-700 hover:bg-zinc-600" },
];

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-12 px-6">
      <header className="text-center space-y-3">
        <h1 className="text-5xl font-bold tracking-tight">Truco RL</h1>
        <p className="text-zinc-400 max-w-xl">
          Treine um agente de aprendizado por reforço para jogar Truco, meça a
          força dele na arena e enfrente-o você mesmo.
        </p>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-lg">
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`px-6 py-4 rounded-xl ${l.tone} transition-colors text-lg font-semibold shadow text-center`}
          >
            {l.label}
          </Link>
        ))}
      </div>
    </main>
  );
}
