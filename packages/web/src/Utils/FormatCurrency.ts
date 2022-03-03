import { noNegZero } from './NoNegZero'

// creating the formatter on server was causing "maximumFractionDigits value is out of range"
export const formatCurrency = (x: number, digits = 0) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    // Without the minimun, maximumFractionDigits of 0 was throwing out of range error.
    // https://stackoverflow.com/a/48562260
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(noNegZero(x))
