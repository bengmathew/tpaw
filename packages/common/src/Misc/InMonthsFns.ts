import { linearFnFromPointAndSlope } from '../Utils'
import { pluralize } from './Pluralize'
import _ from 'lodash'

export namespace InMonthsFns {
  export const getFullYears = ({ inMonths }: { inMonths: number }) =>
    Math.floor(inMonths / 12)

  export const toStr = (x: { inMonths: number }) => {
    const numYears = getFullYears(x)
    const monthsStr = pluralize(x.inMonths % 12, 'month')
    return numYears === 0
      ? monthsStr
      : `${pluralize(numYears, 'year')}, ${monthsStr}`
  }

  export const getFromMFN =
    (x: { inMonths: number }, xAsMFN: number, slope: 1 | -1) => (mfn: number) =>
      _.round(linearFnFromPointAndSlope(xAsMFN, x.inMonths, slope)(mfn))
}
