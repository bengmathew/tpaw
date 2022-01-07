import React from 'react'
import {useSimulation} from '../../App/WithSimulation'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {
  paramsInputValidate,
  paramsInputValidateYearRange,
} from './Helpers/ParamInputValidate'

export const ParamsInputExtraSpending = React.memo(() => {
  const {params} = useSimulation()

  const warn = paramsInputValidate(params, 'extraSpending')

  const defaultYearRange = {
    start: 'retirement' as const,
    end: Math.min(params.age.end, params.age.retirement + 5),
  }

  return (
    <div className="">
      <p className="mb-2">
        If you have extra spending needs during any years of your retirement,
        you can account for that here. For example, travel during early
        retirement, mortgage payments, and kids college tuition.
      </p>
      <p className="mb-2">
        You can categorize your extra spending as essential or discretionary.
        Essential spending will be funded with 100% bonds. Discretionary
        spending will be funded by your regular portfolio.
      </p>
      <ByYearSchedule
        className=""
        type="full"
        heading="Essential"
        addHeading="Add an Essential Expense"
        editHeading="Edit Essential Expense Entry"
        defaultYearRange={defaultYearRange}
        entries={params => params.withdrawals.fundedByBonds}
        validateYearRange={(params, x) =>
          paramsInputValidateYearRange(params, 'extraSpending', x)
        }
      />
      <ByYearSchedule
        className="mt-4"
        type="full"
        heading="Discretionary"
        addHeading="Add a Discretionary Expense"
        editHeading="Edit Discretionary Expense Entry"
        defaultYearRange={defaultYearRange}
        entries={params => params.withdrawals.fundedByRiskPortfolio}
        validateYearRange={(params, x) =>
          paramsInputValidateYearRange(params, 'extraSpending', x)
        }
      />
    </div>
  )
})
