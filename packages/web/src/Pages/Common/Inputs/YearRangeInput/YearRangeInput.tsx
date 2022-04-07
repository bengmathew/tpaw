import React from 'react'
import { Year, YearRange } from '../../../../TPAWSimulator/TPAWParams'
import { pluralize } from '../../../../Utils/Pluralize'
import { useSimulation } from '../../../App/WithSimulation'
import { YearRangeEdgeInput } from './YearRangeEdgeInput'

export type YearRangeInputProps = React.ComponentProps<typeof YearRangeInput>
type YearTypes =
  | 'now'
  | 'lastWorkingYear'
  | 'retirement'
  | 'maxAge'
  | 'numericAge'
  | 'forNumOfYears'
export const YearRangeInput = React.memo(
  ({
    className = '',
    ...props
  }: {
    className?: string
    value: YearRange
    setValue: (year: YearRange) => void
    range: {start: Year; end: Year}
    choices: {start: YearTypes[]; end: YearTypes[]}
  }) => {
    const {value, range} = props
    const {yearRangeBoundsCheck, yearRangeLength} = useSimulation().paramsExt
    const {errorMsgs} = yearRangeBoundsCheck(value, range)

    return (
      <div className={`${className}`}>
        <YearRangeEdgeInput className="" edge="start" {...props} />
        <YearRangeEdgeInput className="" edge="end" {...props} />
        {errorMsgs.length > 0 ? (
          <div className="bg-red-100 rounded-lg p-2 text-errorFG mt-4">
            {errorMsgs.map((x, i) => (
              <h2 key={i}>{x}</h2>
            ))}
          </div>
        ) : (
          <h2 className="mt-1 ">
            {`That's ${pluralize(yearRangeLength(value), 'year')}.`}
          </h2>
        )}
      </div>
    )
  }
)

