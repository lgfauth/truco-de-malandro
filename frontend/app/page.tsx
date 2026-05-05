import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-12 px-6">
      <header className="text-center space-y-3">
        <h1 className="text-5xl font-bold tracking-tight">Truco RL</h1>
        <p className="text-zinc-400 max-w-xl">
          Treine um agente de aprendizado por reforço para jogar Truco e
          enfrente-o você mesmo.
        </p>
      </header>

      <div className="flex flex-col sm:flex-row gap-4">
        <Link
          href="/watch"
          className="px-6 py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 transition-colors text-lg font-semibold shadow"
        >
          Assistir treino
        </Link>
        <Link
          href="/play"
          className="px-6 py-4 rounded-xl bg-amber-600 hover:bg-amber-500 transition-colors text-lg font-semibold shadow"
        >
          Jogar contra a IA
        </Link>
      </div>
    </main>
  );
}
