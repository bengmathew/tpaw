import { Month, MonthRange, noCase } from '@tpaw/common'
import React from 'react'
import { PlanParamsExt } from '../../TPAWSimulator/PlanParamsExt'
import { numMonthsStr } from '../../Utils/NumMonthsStr'
import { SimpleRange } from '../../Utils/SimpleRange'
import { yourOrYourPartners } from '../../Utils/YourOrYourPartners'
import { useSimulation } from '../App/WithSimulation'

export const MonthRangeDisplay = React.memo(
  ({
    className = '',
    value,
    skipLength,
    range,
  }: {
    className?: string
    value: MonthRange
    range: SimpleRange | null
    skipLength: boolean
  }) => {
    const { paramsExt } = useSimulation()
    const { monthRangeBoundsCheck } = paramsExt

    const error =
      range && monthRangeBoundsCheck(value, range).errorMsgs.length > 0
    const monthRangeStr = monthRangeToString(value, paramsExt)

    return (
      <div className={`${className} ${error ? 'text-errorFG' : ''}`}>
        <h2>{monthRangeStr.from.full}</h2>
        <h2>{monthRangeStr.until.full}</h2>
        {!skipLength && monthRangeStr.duration && (
          <h2>{monthRangeStr.duration}</h2>
        )}
      </div>
    )
  },
)

export const monthRangeToString = (
  value: MonthRange,
  paramsExt: PlanParamsExt,
) => {
  const { params, monthRangeLength } = paramsExt
  const { withPartner } = params.people

  let fragments = { start: [] as string[], date: null as string | null }
  const combineFragments = () => {
    const { start: startFragments, date } = fragments
    fragments = { start: [] as string[], date: null }
    const start = startFragments.join(' ')
    return { start, date, full: date === null ? start : `${start} ${date}` }
  }

  let from: ReturnType<typeof combineFragments>
  let until: ReturnType<typeof combineFragments>
  let duration = '' as string | null

  // const yourOrYourPartnersAgeIs = ({
  //   person,
  // }: {
  //   person: 'person1' | 'person2'
  // }) => (person === 'person1' ? 'your age is' :  `your partner's age is`)

  const pushNamedAge = (month: Extract<Month, { type: 'namedAge' }>) => {
    switch (month.age) {
      case 'lastWorkingMonth':
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('last working month')
        break
      case 'retirement':
        if (withPartner) fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('retirement')
        break
      case 'max':
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('max age')
        break
      default:
        noCase(month.age)
    }
  }

  const pushMonthForDuration = (month: Month) => {
    switch (month.type) {
      case 'now':
        fragments.start.push('now')
        break
      case 'namedAge':
        fragments.start.push('at')
        pushNamedAge(month)
        break
      case 'numericAge':
        // if (withPartner) {
        fragments.start.push('when')
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('age is')
        fragments.date = numMonthsStr(month.ageInMonths)
        // } else {
        //   fragments.start.push(`at age`)
        //   fragments.date = numMonthsStr(month.ageInMonths)
        // }
        break
      default:
        noCase(month)
    }
  }

  switch (value.type) {
    case 'startAndEnd':
      fragments.start.push('From')
      switch (value.start.type) {
        case 'now':
          fragments.start.push('now')
          break
        case 'namedAge':
          pushNamedAge(value.start)
          break
        case 'numericAge':
          // if (withPartner) {
          fragments.start.push('when')
          fragments.start.push(yourOrYourPartners(value.start))
          fragments.start.push('age is')
          fragments.date = numMonthsStr(value.start.ageInMonths)
          // } else {
          //     fragments.start.push(`age`)
          //     fragments.date = numMonthsStr(value.start.ageInMonths)
          // }
          break
        default:
          noCase(value.start)
      }
      from = combineFragments()

      fragments.start.push('Until')
      switch (value.end.type) {
        case 'now':
          fragments.start.push('now')
          break
        case 'namedAge':
          pushNamedAge(value.end)
          break
        case 'numericAge':
          // if (withPartner) {
          fragments.start.push(yourOrYourPartners(value.end))
          fragments.start.push('age is')
          fragments.date = numMonthsStr(value.end.ageInMonths)
          // } else {
          //     fragments.start.push(`age`)
          //     fragments.date = numMonthsStr(value.end.ageInMonths)
          // }
          break
        default:
          noCase(value.end)
      }
      until = combineFragments()
      duration = `That's ${numMonthsStr(monthRangeLength(value))}`
      break
    case 'startAndNumMonths':
      fragments.start.push('Starting')
      pushMonthForDuration(value.start)
      from = combineFragments()
      fragments.start.push('For a period of')
      fragments.date = numMonthsStr(value.numMonths)
      until = combineFragments()
      break
    case 'endAndNumMonths':
      fragments.start.push('For a period of')
      fragments.date = numMonthsStr(value.numMonths)
      from = combineFragments()

      fragments.start.push('Ending')
      pushMonthForDuration(value.end)
      until = combineFragments()

      break
    default:
      noCase(value)
  }
  return { from, until, duration }
}
