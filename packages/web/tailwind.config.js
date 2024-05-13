const colors = require('tailwindcss/colors')
delete colors.lightBlue
delete colors.warmGray
delete colors.trueGray
delete colors.blueGray
delete colors.coolGray

module.exports = {
  content: ['./pages/**/*.{js,ts,jsx,tsx}', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'media',
  theme: {
    fontFamily: {
      font1: ['Montserrat', 'sans-serif'],
      font2: ['Karla', 'sans-serif'],
      mono: ['Roboto Mono', 'monospace'],
    },

    colors: {
      pageBG: 'white',
      planBG: colors.gray[100],
      // planBG: colors.blue[50],
      // planBG: '',
      // chartBG: '#f7dacd',
      chartBG: colors.gray[400],
      pageFG: colors.gray[800],
      cardBG: 'rgba(255,255,255, .95)',
      pageFGLight: colors.gray[600],
      boldFG: colors.indigo[800],
      alt: colors.blue[900],
      darkGray: colors.gray[800],

      // Error and Success
      errorFG: colors.red[500],
      errorBlockBG: colors.red[500],
      errorBlockFG: colors.gray[100],
      successFG: colors.green[500],
      successDarkFG: colors.green[800],
      successBlockBG: colors.green[500],
      successBlockFG: colors.gray[100],
      amberFG: colors.yellow[600],
      // theme2: '#f7dacd',
      // theme2: '#9ed4cf',
      // theme2:colors.gray[300],
      theme1: colors.teal[600],
      theme1Dark: colors.teal[700],

      // Interesting colors
      intersting1: `rgb(245, 247, 252)`,

      ...colors,
    },
    extend: {
      screens: {
        learn: '900px',
      },
    },
  },

  plugins: [
    require('@headlessui/tailwindcss'),
    // From: https://github.com/tailwindlabs/tailwindcss-intellisense/issues/227
    ({ addUtilities }) => {
      addUtilities({
        '.lighten': {
          '@apply opacity-70': {},
        },
        '.lighten-2': {
          '@apply opacity-50': {},
        },
        // Modal Dialog
        '.dialog-outer-div': {
          maxWidth: 'min(800px, calc(100vw - 20px))',
          minWidth: 'min(400px, calc(100vw - 20px))',
        },
        '.dialog-heading': {
          '@apply font-bold text-lg': {},
        },
        '.dialog-content-div': {
          '@apply mt-4': {},
        },
        '.dialog-button-div': {
          '@apply flex justify-end mt-6 gap-x-4': {},
        },
        '.dialog-button-warning': {
          '@apply btn2-warning btn2-md': {},
        },
        '.dialog-button-dark': {
          '@apply btn2-dark btn2-md': {},
        },
        '.dialog-button-cancel': {
          '@apply btn2-md disabled:lighten-2 -mr-2': {},
        },

        // ---- Context Menu ----
        '.context-menu-outer-div': {
          '@apply py-2.5 rounded-lg': {},
        },
        '.context-menu-section': {
          '@apply mt-5 first:mt-0': {},
        },
        '.context-menu-section-heading': {
          '@apply mx-4 pt-1 pb-0.5 mb-1 lighten-2 text-sm font-medium text-left':
            {},
        },
        '.context-menu-item': {
          '@apply block px-4 py-1.5 text-start ui-active:bg-gray-200 w-full':
            {},
        },
        '.context-menu-icon': {
          '@apply inline-block w-[25px] mr-1 text-center': {},
        },

        // ---- Button ----
        '.btn2-warning': {
          '@apply text-white bg-errorBlockBG disabled:lighten-2': {},
        },
        '.btn2-dark': {
          '@apply text-white bg-darkGray disabled:lighten-2': {},
        },
        '.btn2-xs': {
          '@apply rounded-full py-0.5 px-3 text-sm': {},
        },
        '.btn2-sm': {
          '@apply rounded-full py-0.5 px-4 text-base': {},
        },
        '.btn2-md': {
          '@apply rounded-full py-1 px-4 text-lg': {},
        },
        '.btn2-lg': {
          '@apply rounded-full py-2 px-6 text-lg': {},
        },

        '.custom-shadow-md': {
          boxShadow: '0 4px 14px 0 rgba(0, 0, 0, 0.1)',
        },
        '.custom-shadow-md-dark': {
          boxShadow: '0 4px 14px 0 rgba(0, 0, 0, 0.3)',
        },
      })
    },
  ],
}
