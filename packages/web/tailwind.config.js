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
      pageFG: colors.gray[800],
      cardBG: 'rgba(255,255,255, .9)',
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

      ...colors,
    },
    extend: {
      screens: {
        learn: '900px',
      },
    },
  },

  plugins: [],
}
