import clsx from 'clsx'
import React from 'react'
import {
  NormalizedGlidePath,
  NormalizedGlidePathEntry,
} from '../../../../../UseSimulator/NormalizePlanParams/NormalizeGlidePath'
import { getNormalizedMonthStr } from '../../../../Common/MonthOrDurationDisplay'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTurnDownLeft } from '@fortawesome/pro-solid-svg-icons'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'

export const PlanInputSummaryGlidePath = React.memo(
  ({
    className = '',
    normValue,
  }: {
    className?: string
    normValue: NormalizedGlidePath
  }) => {
    const { now, intermediate, end, atOrPastEnd } = normValue
    const format = formatPercentage(0)
    return (
      <div
        className={`${className} inline-grid gap-x-10 items-center`}
        style={{ grid: 'auto/auto auto' }}
      >
        <h2>Now</h2>
        <h2 className="text-right">{format(now.stocks)}</h2>
        <_Intermediate intermediate={intermediate} format={format} />
        <h2>Max age</h2>
        <h2 className="text-right">{format(end.stocks)}</h2>
        <_Intermediate intermediate={atOrPastEnd} format={format} />
      </div>
    )
  },
)

const _Intermediate = React.memo(
  ({
    intermediate,
    format,
  }: {
    intermediate: NormalizedGlidePathEntry[]
    format: (x: number) => string
  }) => {
    return (
      <>
        {intermediate.map((x, i) => (
          <React.Fragment key={i}>
            <h2 className={clsx(x.month.errorMsg && 'text-errorFG')}>
              {getNormalizedMonthStr(x.month).full}
            </h2>
            <h2 className="text-right">{format(x.stocks)}</h2>
            {x.month.errorMsg && (
              <h2 className="text-errorFG col-span-2 pl-2">
                <FontAwesomeIcon
                  className="fa-rotate-90 mb-1 mr-1 text-xs"
                  icon={faTurnDownLeft}
                />{' '}
                {x.month.errorMsg}
              </h2>
            )}
          </React.Fragment>
        ))}
      </>
    )
  },
)
