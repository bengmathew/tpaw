import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {useSimulation} from '../../App/WithSimulation'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {
  paramsInputValidate,
  paramsInputValidateYearRange,
} from './Helpers/ParamInputValidate'

export const ParamsInputExtraSpending = React.memo(() => {
  const {params} = useSimulation()
  const content = usePlanContent()
  const warn = paramsInputValidate(params, 'extraSpending')

  const defaultYearRange = {
    start: 'retirement' as const,
    end: Math.min(params.age.end, params.age.retirement + 5),
  }

  return (
    <div className="">
      <Contentful.RichText
        body={content.extraSpending.intro.fields.body}
        p="mb-6"
      />
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
        className=""
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
