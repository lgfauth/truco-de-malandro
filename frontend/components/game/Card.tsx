"use client";

import { memo } from "react";

import type { CardDTO } from "@/types/game";

export type CardSize = "hand" | "table" | "vira" | "back" | "mini";

const SUIT_GLYPH = ["♦", "♠", "♥", "♣"];
export const RANK_LABEL: Record<number, string> = {
  4: "4", 5: "5", 6: "6", 7: "7",
  10: "Q", 11: "J", 12: "K", 13: "A",
  14: "2", 15: "3",
};
// Cyclic order used to find the manilha: the rank right after the vira.
const RANK_CYCLE = [4, 5, 6, 7, 10, 11, 12, 13, 14, 15];

export function manilhaRank(vira: CardDTO | undefined): number | null {
  if (!vira) return null;
  const i = RANK_CYCLE.indexOf(vira.rank);
  return i < 0 ? null : RANK_CYCLE[(i + 1) % RANK_CYCLE.length];
}

function PlayingCardImpl({
  card,
  size,
  manilha = false,
  className = "",
}: {
  card: CardDTO;
  size: CardSize;
  manilha?: boolean;
  className?: string;
}) {
  const suit = SUIT_GLYPH[card.suit] ?? "?";
  const red = card.suit === 0 || card.suit === 2; // ouros, copas
  const rank = RANK_LABEL[card.rank] ?? String(card.rank);
  return (
    <div
      className={`pcard pcard-${size} relative rounded-[9%/6.5%] bg-white select-none shadow-[0_2px_6px_rgba(0,0,0,0.35)] ${
        manilha ? "ring-[3px] ring-amber-400" : "ring-1 ring-black/10"
      } ${className}`}
      style={{ color: red ? "#c81e1e" : "#18181b" }}
      aria-label={`${rank} de ${["ouros", "espadas", "copas", "paus"][card.suit]}${manilha ? ", manilha" : ""}`}
      role="img"
    >
      <div className="absolute top-[5%] left-[8%] flex flex-col items-center leading-none font-bold">
        <span className="text-[26cqw]">{rank}</span>
        <span className="text-[20cqw] -mt-[2cqw]">{suit}</span>
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-[52cqw] leading-none translate-y-[6%]">{suit}</span>
      </div>
      <div className="absolute bottom-[5%] right-[8%] flex flex-col items-center leading-none font-bold rotate-180">
        <span className="text-[26cqw]">{rank}</span>
        <span className="text-[20cqw] -mt-[2cqw]">{suit}</span>
      </div>
      {manilha && (
        <span
          className="absolute -top-[6%] -right-[6%] grid place-items-center rounded-full bg-amber-400 text-zinc-900 font-black shadow w-[30cqw] h-[30cqw] text-[18cqw] leading-none"
          aria-hidden
        >
          ★
        </span>
      )}
    </div>
  );
}

export const PlayingCard = memo(PlayingCardImpl);

function CardBackImpl({ size, className = "" }: { size: CardSize; className?: string }) {
  return (
    <div
      className={`pcard pcard-${size} rounded-[9%/6.5%] ring-1 ring-white/40 shadow-[0_2px_6px_rgba(0,0,0,0.35)] ${className}`}
      style={{
        backgroundColor: "#1e3a8a",
        backgroundImage:
          "repeating-linear-gradient(45deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 9px), repeating-linear-gradient(-45deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 9px)",
      }}
      aria-hidden
    />
  );
}

export const CardBack = memo(CardBackImpl);

/** Dashed outline where a card will land. */
export function CardSlot({ size, label }: { size: CardSize; label?: string }) {
  return (
    <div
      className={`pcard pcard-${size} rounded-[9%/6.5%] border-2 border-dashed border-white/20 grid place-items-center text-[11px] text-white/40 text-center px-1`}
    >
      {label}
    </div>
  );
}
