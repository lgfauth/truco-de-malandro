import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Truco RL",
  description: "Reinforcement learning agent that plays Truco",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {children}
      </body>
    </html>
  );
}
