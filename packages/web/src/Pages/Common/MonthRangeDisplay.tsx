import { block, noCase } from '@tpaw/common'
import clsx from 'clsx'
import React from 'react'
import { NormalizedMonthRange } from '@tpaw/common'
import { MonthOrDurationDisplay } from './MonthOrDurationDisplay'
import { InMonthsFns } from '@tpaw/common'

export const MonthRangeDisplay = React.memo(
  ({
    className,
    value,
  }: {
    className?: string
    value: NormalizedMonthRange
  }) => {
    return (
      <div className={clsx(className)}>
        {block(() => {
          switch (value.type) {
            case 'startAndEnd':
              const durationStr = getMonthRangeDurationStr(value)
              return (
                <>
                  <div className="flex gap-x-2">
                    <h2 className="">Start: </h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.start}
                    />
                  </div>
                  <div className="flex gap-x-2">
                    <h2 className="">End: </h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.end}
                    />
                  </div>
                  {durationStr && <h2 className="">{durationStr}</h2>}
                </>
              )
            case 'startAndDuration':
              return (
                <>
                  <div className="flex gap-x-2">
                    <h2 className="">Start: </h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.start}
                    />
                  </div>
                  <div className="flex gap-x-2">
                    <h2 className="">Duration:</h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.duration}
                    />
                  </div>
                </>
              )
            case 'endAndDuration':
              return (
                <>
                  <div className="flex gap-x-2">
                    <h2 className="">End: </h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.end}
                    />
                  </div>
                  <div className="flex gap-x-2">
                    <h2 className="">Duration:</h2>
                    <MonthOrDurationDisplay
                      className=""
                      normValue={value.duration}
                    />
                  </div>
                </>
              )
            default:
              noCase(value)
          }
        })}
      </div>
    )
  },
)

export const getMonthRangeDurationStr = (
  value: Extract<NormalizedMonthRange, { type: 'startAndEnd' }>,
) =>
  value.end.isInThePast || value.end.asMFN < value.start.asMFN
    ? null
    : `That's ${InMonthsFns.toStr({
        inMonths: value.end.asMFN - value.start.asMFN + 1,
      })}`
