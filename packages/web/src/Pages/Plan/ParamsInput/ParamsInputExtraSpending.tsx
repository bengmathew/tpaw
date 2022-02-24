import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {useSimulation} from '../../App/WithSimulation'
import {
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingEssentialType,
} from '../ChartPanel/ChartPanelType'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {
  paramsInputValidate,
  paramsInputValidateYearRange,
} from './Helpers/ParamInputValidate'

export const ParamsInputExtraSpending = React.memo(
  ({
    chartType,
    setChartType,
  }: {
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  }) => {
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
          p="mb-6 p-base"
        />
        <ByYearSchedule
          className="mb-6"
          type="full"
          heading="Essential"
          addHeading="Add an Essential Expense"
          editHeading="Edit Essential Expense Entry"
          defaultYearRange={defaultYearRange}
          entries={params => params.withdrawals.fundedByBonds}
          onBeforeDelete={id => {
            if (isChartPanelSpendingEssentialType(chartType)) {
              const currChartID = chartPanelSpendingEssentialTypeID(chartType)
              if (id === currChartID) {
                setChartType('spending-total')
              }
            }
          }}
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
  }
)
