const { fontFamily } = require("tailwindcss/defaultTheme");

const generateColorClass = (variable) => {
  return ({ opacityValue }) =>
    opacityValue
      ? `rgba(var(--${variable}), ${opacityValue})`
      : `rgb(var(--${variable}))`
}

const textColor = {
  primary: generateColorClass('text-primary'),
}

const backgroundColor = {
  primary: generateColorClass('bg-primary'),
  secondary: generateColorClass('bg-secondary'),
  tertiary: generateColorClass('bg-tertiary'),
}

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./app/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  corePlugins: {
    preflight: false,
  },
  theme: {
    extend: {
      textColor,
      backgroundColor,
      colors: {
        primary: generateColorClass('primary'),
        bg: {
          ...backgroundColor
        },
        text: {
          ...textColor
        },
      },
    },
  },
  keyframes: {
    blink: {
      "0%, 100%": { opacity: 1 },
      "50%": { opacity: 0 },
    },
  },
  plugins: [
    require('tailwind-scrollbar')({ nocompatible: true }),
    require('@tailwindcss/typography'),
    require("@tailwindcss/line-clamp"),
  ],
};
