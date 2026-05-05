import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        felt: {
          900: "#0b3d1f",
          800: "#0e4a26",
          700: "#125c30",
        },
      },
    },
  },
  plugins: [],
};

export default config;
