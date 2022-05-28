import {TPAWParams, Year} from '../../../../TPAWSimulator/TPAWParams'
import {noCase} from '../../../../Utils/Utils'
import {YearInputProps} from './YearInput'

export function yearLabel(
  value: Year | {numYears: number},
  location: YearInputProps['location'],
  type: 'includeNumber' | 'numberIsBlank',
  {people}: TPAWParams
) {
  if ('numYears' in value) {
    return type === 'includeNumber'
      ? `for ${value.numYears} years`
      : 'for number of years'
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
    case 'numericAge':
      result.push(
        value.person === 'person1'
          ? people.withPartner
            ? `your`
            : ''
          : value.person === 'person2'
          ? `your partner's`
          : noCase(value.person)
      )
      result.push(
        value.age === 'lastWorkingYear'
          ? `last working year`
          : value.age === 'retirement'
          ? `retirement`
          : value.age === 'max'
          ? `max age`
          : typeof value.age === 'number'
          ? type === 'includeNumber'
            ? `age ${value.age}`
            : 'age'
          : noCase(value.age)
      )
      break
    default:
      noCase(value)
  }
  return result.join(' ').trim()
}
