import { PlanParams } from '@tpaw/common'
import { planResultsChartLabel } from '../../../Plan/PlanResults/PlanResultsChartCard/PlanResultsChartLabel'
import {
  PlanResultsChartType,
  getPlanResultsChartSpendingTotalFundingSourcesPercentile,
  isPlanResultsChartSpendingTotalFundingSourcesType,
} from '../../../Plan/PlanResults/PlanResultsChartType'
import { PlanParamsNormalized } from '../../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'

export const getPlanPrintChartLabel = (
  planParamsNorm: PlanParamsNormalized,
  type: PlanResultsChartType,
) => {
  const {
    label: labelIn,
    subLabel: subLabelIn,
    yAxisDescription,
  } = planResultsChartLabel(planParamsNorm, type)
  const label = labelIn
  const subLabel = isPlanResultsChartSpendingTotalFundingSourcesType(type)
    ? `Funding Sources for the ${getPlanResultsChartSpendingTotalFundingSourcesPercentile(
        type,
      )}th Percentile`
    : subLabelIn
  const yAxisDescriptionStr = yAxisDescription
    ? yAxisDescription.notMobile.map((x) => x.value).join(' ')
    : null
  const description =
    type === 'spending-total-funding-sources-5' ? (
      <span>
        The graph also shows the 50
        <span className=" align-super text-[10px]">th</span> and 95
        <span className=" align-super text-[10px]">th</span> percentiles of
        spending in the background.
      </span>
    ) : (
      <span>
        The graph shows the 5
        <span className=" align-super text-[10px]">th</span> to 95
        <span className=" align-super text-[10px]">th</span> percentile range
        with the 50<span className=" align-super text-[10px]">th</span>{' '}
        percentile in bold.
      </span>
    )

  return { label, subLabel, yAxisDescriptionStr, description }
}
