"use client";

// ---------------------------------------------------------------------------
// Inline SVG card helper (used inside RulesModal)
// ---------------------------------------------------------------------------
function SvgCard({
  rank,
  suit,
  x,
  y,
}: {
  rank: string;
  suit: string;
  x: number;
  y: number;
}) {
  const isRed = suit === "♥" || suit === "♦";
  const color = isRed ? "#dc2626" : "#1a1a1a";
  return (
    <g transform={`translate(${x},${y})`}>
      <rect width={40} height={58} rx={6} fill="#fff" stroke="#d4d4d8" strokeWidth={1} />
      <text x={4} y={13} fontSize={10} fontWeight={700} fill={color}>{rank}</text>
      <text x={4} y={23} fontSize={10} fill={color}>{suit}</text>
      <text x={20} y={36} fontSize={16} textAnchor="middle" fill={color}>{suit}</text>
      <text x={36} y={53} fontSize={10} fontWeight={700} fill={color} textAnchor="middle" transform="rotate(180,36,48)">{rank}</text>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Rules Modal
// ---------------------------------------------------------------------------
export function RulesModal({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 bg-black/70 z-50 overflow-y-auto"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="max-w-lg mx-4 sm:mx-auto my-6 bg-zinc-900 rounded-2xl p-5 sm:p-6 text-zinc-100 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold">📖 Regras do Truco Paulista</h2>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-zinc-100 text-2xl leading-none"
            aria-label="Fechar"
          >
            ×
          </button>
        </div>

        {/* Seção 1 — Objetivo */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Objetivo</h3>
          <p className="text-sm text-zinc-300">
            Chegar a <strong>12 pontos</strong> antes da IA. Cada mão vale pontos conforme o stake negociado.
          </p>
        </section>

        {/* Seção 2 — Hierarquia */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-2">Hierarquia das cartas</h3>
          <div className="overflow-x-auto">
            <svg width={460} height={80} viewBox="0 0 460 80" className="block">
              {[
                ["4","♠"],["5","♠"],["6","♠"],["7","♠"],
                ["Q","♠"],["J","♠"],["K","♠"],["A","♠"],
                ["2","♠"],["3","♠"],
              ].map(([r, s], i) => (
                <SvgCard key={i} rank={r} suit={s} x={i * 46} y={0} />
              ))}
            </svg>
          </div>
          <p className="text-xs text-zinc-400 mt-1">da mais fraca para a mais forte</p>
        </section>

        {/* Seção 3 — Manilhas */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Manilhas</h3>
          <p className="text-sm text-zinc-300 mb-3">
            A vira define a manilha — a carta seguinte na ordem vira manilha. Ex: se a vira for o 6, a manilha é o 7.
          </p>
          <div className="overflow-x-auto">
            <svg width={300} height={90} viewBox="0 0 300 90" className="block">
              {/* Vira */}
              <SvgCard rank="6" suit="♠" x={0} y={0} />
              {/* Arrow */}
              <text x={50} y={34} fontSize={18} fill="#a1a1aa">→</text>
              {/* 4 manilhas */}
              {[["7","♣","#1"],["7","♥","#2"],["7","♠","#3"],["7","♦","#4"]].map(([r,s,b],i) => (
                <g key={i}>
                  <SvgCard rank={r} suit={s} x={72 + i * 58} y={0} />
                  <text
                    x={72 + i * 58 + 20}
                    y={75}
                    fontSize={9}
                    textAnchor="middle"
                    fill="#a1a1aa"
                  >{b}</text>
                </g>
              ))}
            </svg>
          </div>
        </section>

        {/* Rodadas e empates */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Rodadas e empates</h3>
          <ul className="text-sm text-zinc-300 list-disc pl-5 space-y-1">
            <li>A mão é melhor de 3 rodadas; quem vence a rodada começa a próxima.</li>
            <li>Empatou a 1ª: quem vencer a próxima rodada leva a mão.</li>
            <li>Venceu a 1ª e depois empatou: quem venceu a 1ª leva a mão.</li>
            <li>Três empates: ninguém pontua.</li>
          </ul>
        </section>

        {/* Seção 4 — Truco */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Truco</h3>
          <p className="text-sm text-zinc-300 mb-3">
            Qualquer jogador pode pedir Truco. O adversário pode aceitar, correr ou aumentar. Stake sobe: 1 → 3 → 6 → 9 → 12.
          </p>
          <svg width={280} height={36} viewBox="0 0 280 36" className="block">
            {[1,3,6,9,12].map((v, i) => (
              <g key={v}>
                <rect x={i * 56} y={4} width={42} height={28} rx={6} fill="#3f3f46" stroke="#52525b" />
                <text x={i * 56 + 21} y={23} fontSize={13} fontWeight={700} textAnchor="middle" fill="#f4f4f5">{v}</text>
                {i < 4 && (
                  <text x={i * 56 + 47} y={22} fontSize={14} fill="#a1a1aa">→</text>
                )}
              </g>
            ))}
          </svg>
        </section>

        {/* Seção 5 — Mão de 11 */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Mão de 11</h3>
          <p className="text-sm text-zinc-300 mb-3">
            Quando um jogador chega a 11 pontos, ele olha as próprias cartas e decide antes de jogar: aceitar (a mão vale 3) ou correr (o adversário ganha 1 ponto). Não se pode pedir truco nessa mão.
          </p>
          <svg width={260} height={80} viewBox="0 0 260 80" className="block">
            <SvgCard rank="A" suit="♠" x={0} y={10} />
            <SvgCard rank="3" suit="♥" x={46} y={10} />
            <SvgCard rank="7" suit="♣" x={92} y={10} />
            <text x={148} y={30} fontSize={11} fill="#a1a1aa">→</text>
            <rect x={162} y={8} width={44} height={22} rx={5} fill="#059669" />
            <text x={184} y={23} fontSize={10} fontWeight={700} textAnchor="middle" fill="#fff">Aceitar</text>
            <rect x={162} y={38} width={44} height={22} rx={5} fill="#be123c" />
            <text x={184} y={53} fontSize={10} fontWeight={700} textAnchor="middle" fill="#fff">Correr</text>
          </svg>
        </section>

        {/* Seção 6 — Mão de Ferro */}
        <section>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-400 mb-1">Mão de Ferro</h3>
          <p className="text-sm text-zinc-300">
            Quando <strong>os dois jogadores</strong> têm 11 pontos, não é possível pedir truco e a mão vale <strong>3 pontos</strong>. Você joga sem ver as suas cartas — a IA, malandra, vê as dela.
          </p>
        </section>

        <button
          onClick={onClose}
          className="w-full py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-sm font-semibold"
        >
          Fechar
        </button>
      </div>
    </div>
  );
}
