import { assert, assertFalse, block } from '@tpaw/common'
import React from 'react'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { trimAndNullify } from '../../Utils/TrimAndNullify'
import { MonthRangeDisplay } from './MonthRangeDisplay'
import { NormalizedLabeledAmountTimed } from '@tpaw/common'

export const LabeledAmountTimedDisplay = React.memo(
  ({
    className = '',
    labelClassName = '',
    entry,
  }: {
    className?: string
    labelClassName?: string
    entry: NormalizedLabeledAmountTimed
  }) => {
    const { amountAndTiming } = entry
    return (
      <div className={`${className}`}>
        <h2 className={`${labelClassName}`}>
          {trimAndNullify(entry.label) ?? '<no label>'}
        </h2>
        {block(() => {
          switch (amountAndTiming.type) {
            case 'inThePast': {
              return (
                <>
                  <h2 className="">Completed: This entry is in the past</h2>
                </>
              )
            }
            case 'oneTime': {
              assertFalse()
            }
            case 'recurring': {
              assert(amountAndTiming.delta === null)
              assert(amountAndTiming.everyXMonths === 1)
              return (
                <>
                  <h2>{`${formatCurrency(amountAndTiming.baseAmount)} per month ${
                    entry.nominal ? '(nominal dollars)' : '(real dollars)'
                  }`}</h2>
                  <MonthRangeDisplay
                    className=""
                    value={amountAndTiming.monthRange}
                  />
                </>
              )
            }
          }
        })}
      </div>
    )
  },
)
