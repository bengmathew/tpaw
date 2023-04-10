import _ from 'lodash'
import { getPrecision } from './GetPrecision'
import { noNegZero } from './NoNegZero'

// creating the formatter on server was causing "maximumFractionDigits value is out of range"
export const formatPercentage = (precision: number | 'full') => (x: number) => {
  return precision === 'full'
    ? `${_.round(x * 100, getPrecision(x) - 2)}%`
    : `${(noNegZero(x) * 100).toFixed(precision)}%`
}
