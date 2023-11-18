import { PlanParams } from '@tpaw/common'
import { planResultsChartLabel } from '../PlanResults/PlanResultsChartCard/PlanResultsChartLabel'
import { PlanResultsChartType } from '../PlanResults/PlanResultsChartType'

export const getPlanPrintChartLabel = (
  planParams: PlanParams,
  type: PlanResultsChartType,
) => {
  const {
    label: labelIn,
    subLabel: subLabelIn,
    yAxisDescription,
  } = planResultsChartLabel(planParams, type)
  const label = labelIn
  const subLabel =
    type === 'spending-total-funding-sources-5'
      ? 'Funding Sources for the 5th Percentile'
      : subLabelIn
  const yAxisDescriptionStr = yAxisDescription
    ? yAxisDescription.notMobile.map((x) => x.value).join(' ')
    : null

  return { label, subLabel, yAxisDescriptionStr }
}
