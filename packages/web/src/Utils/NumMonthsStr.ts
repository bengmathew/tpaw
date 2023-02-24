import { numFullYearsIn } from './NumFullYearsIn'
import { pluralize } from './Pluralize'

export const numMonthsStr = (numMonths: number) =>
  `${pluralize(numFullYearsIn(numMonths), 'year')}, ${pluralize(
    numMonths % 12,
    'month',
  )}`
