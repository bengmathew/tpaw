import React from 'react'
import {Year, YearRange} from '../../TPAWSimulator/TPAWParams'
import {noCase} from '../../Utils/Utils'
import {useSimulation} from '../App/WithSimulation'

export const YearRangeDisplay = React.memo(
  ({
    value,
    range,
    lengthInBrackets,
  }: {
    value: YearRange
    range: {start: Year; end: Year} | null
    lengthInBrackets: boolean
  }) => {
    const {yearRangeToStr, yearRangeBoundsCheck} = useSimulation().paramsExt
    const parts = yearRangeToStr(value, {lengthInBrackets})
    console.dir(parts)
    if (!range) return <span>{parts.map(x => x.value).join(' ')}</span>

    const boundsCheck = yearRangeBoundsCheck(value, range)
    return (
      <>
        {parts.map(({type, value}, i) => {
          const end =
            i === parts.length - 1 || parts[i + 1].value.startsWith('.')
              ? ''
              : ' '
          switch (type) {
            case 'other':
              return (
                <span key={i}>
                  {value}
                  {end}
                </span>
              )
            default:
              const check = boundsCheck[type]
              switch (check) {
                case 'ok':
                  return (
                    <span key={i}>
                      {value}
                      {end}
                    </span>
                  )
                case 'endBeforeStart':
                case 'outOfBounds':
                  return (
                    <span key={i} className="text-errorFG">
                      {value}
                      {end}
                    </span>
                  )
                default:
                  noCase(check)
              }
          }
        })}
      </>
    )
  }
)
