import { ValueForMonthRange } from '@tpaw/common'
import React from 'react'
import { PlanParamsExtended } from '../../UseSimulator/ExtentPlanParams'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { SimpleRange } from '../../Utils/SimpleRange'
import { trimAndNullify } from '../../Utils/TrimAndNullify'
import { MonthRangeDisplay } from './MonthRangeDisplay'

export const ValueForMonthRangeDisplay = React.memo(
  ({
    className = '',
    labelClassName = '',
    entry,
    rangeAsMFN,
    skipLength,
    planParamsExt,
  }: {
    className?: string
    labelClassName?: string
    entry: ValueForMonthRange
    rangeAsMFN: SimpleRange | null
    skipLength: boolean
    planParamsExt: PlanParamsExtended
  }) => {
    const { clampMonthRangeToNow } = planParamsExt
    return (
      <div className={`${className}`}>
        <h2 className={`${labelClassName}`}>
          {trimAndNullify(entry.label) ?? '<no label>'}
        </h2>
        <h2>{`${formatCurrency(entry.value)} per month ${
          entry.nominal ? '(nominal dollars)' : '(real dollars)'
        }`}</h2>
        <MonthRangeDisplay
          className=""
          valueClamped={clampMonthRangeToNow(entry.monthRange)}
          range={rangeAsMFN}
          skipLength={skipLength}
          planParamsExt={planParamsExt}
        />
      </div>
    )
  },
)
