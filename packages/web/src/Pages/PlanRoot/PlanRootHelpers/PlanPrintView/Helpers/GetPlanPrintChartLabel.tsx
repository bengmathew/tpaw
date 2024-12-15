import { PlanParams } from '@tpaw/common'
import { planResultsChartLabel } from '../../../Plan/PlanResults/PlanResultsChartCard/PlanResultsChartLabel'
import {
  PlanResultsChartType,
  getPlanResultsChartSpendingTotalFundingSourcesPercentile,
  isPlanResultsChartSpendingTotalFundingSourcesType,
} from '../../../Plan/PlanResults/PlanResultsChartType'
import { PlanParamsNormalized } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParams'

export const getPlanPrintChartLabel = (
  planParamsNormOfResult: PlanParamsNormalized,
  percentilesOfResult: { low: number; mid: number; high: number },
  type: PlanResultsChartType,
) => {
  const {
    label: labelIn,
    subLabel: subLabelIn,
    yAxisDescription,
  } = planResultsChartLabel(planParamsNormOfResult, type)
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
    type === 'spending-total-funding-sources-low' ? (
      <span>
        The graph also shows the {percentilesOfResult.mid}
        <span className=" align-super text-[10px]">th</span> and{' '}
        {percentilesOfResult.high}
        <span className=" align-super text-[10px]">th</span> percentiles of
        spending in the background.
      </span>
    ) : (
      <span>
        The graph shows the {percentilesOfResult.low}
        <span className=" align-super text-[10px]">th</span> to{' '}
        {percentilesOfResult.high}
        <span className=" align-super text-[10px]">th</span> percentile range
        with the {percentilesOfResult.mid}
        <span className=" align-super text-[10px]">th</span> percentile in bold.
      </span>
    )

  return { label, subLabel, yAxisDescriptionStr, description }
}
