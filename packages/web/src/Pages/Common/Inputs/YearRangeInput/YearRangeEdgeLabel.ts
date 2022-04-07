import {TPAWParams, YearRange} from '../../../../TPAWSimulator/TPAWParams'
import {assert, noCase} from '../../../../Utils/Utils'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'

export function yearRangeEdgeLabel(
  range: YearRange,
  edge: 'start' | 'end',
  type: 'includeNumber' | 'numberIsBlank',
  {people}: TPAWParams
) {
  const value = extendTPAWParams.yearRangeEdge(range, edge)
  if (!value) {
    assert('numYears' in range)
    return type === 'includeNumber'
      ? `for ${range.numYears} years`
      : 'for number of years'
  }
  const result = [] as string[]
  result.push(edge === 'start' ? 'from' : 'until')
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
  return result.join(' ')
}
