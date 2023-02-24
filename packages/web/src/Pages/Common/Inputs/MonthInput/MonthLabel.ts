import { Month, PlanParams } from '@tpaw/common'
import { numMonthsStr } from '../../../../Utils/NumMonthsStr'
import { noCase } from '../../../../Utils/Utils'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { MonthInputProps } from './MonthInput'

export function monthLabel(
  value: Month | { numMonths: number },
  location: MonthInputProps['location'],
  type: 'includeNumber' | 'numberIsBlank',
  { people }: PlanParams,
) {
  if ('numMonths' in value) {
    return type === 'includeNumber'
      ? `for ${numMonthsStr(value.numMonths)}`
      : 'for a period of'
  }
  const result = [] as string[]

  location === 'rangeStart'
    ? result.push('from')
    : location === 'rangeEnd'
    ? result.push('until')
    : location === 'standalone'
    ? ''
    : noCase(location)

  switch (value.type) {
    case 'now':
      result.push('now')
      break
    case 'namedAge':
      result.push(people.withPartner ? yourOrYourPartners(value.person) : '')
      result.push(
        value.age === 'lastWorkingMonth'
          ? `last working month`
          : value.age === 'retirement'
          ? `retirement`
          : value.age === 'max'
          ? `max age`
          : noCase(value.age),
      )
      break
    case 'numericAge':
      result.push(people.withPartner ? yourOrYourPartners(value.person) : '')
      result.push(
        type === 'includeNumber'
          ? `age ${numMonthsStr(value.ageInMonths)}`
          : 'age',
      )
      break
    default:
      noCase(value)
  }
  return result.join(' ').trim()
}
