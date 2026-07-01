/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      colors: {
        mint: {
          50:  "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#10b981",
          700: "#059669",
          800: "#047857",
          900: "#065f46",
        },
      },
      boxShadow: {
        mint: "0 1px 3px rgba(16,185,129,0.08), 0 4px 16px rgba(16,185,129,0.06)",
        "mint-md": "0 4px 24px rgba(16,185,129,0.12)",
      },
    },
  },
  plugins: [],
};
