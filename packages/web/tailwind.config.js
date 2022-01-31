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
      pageFG: colors.gray[800],
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

      theme1: colors.teal[600],

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
