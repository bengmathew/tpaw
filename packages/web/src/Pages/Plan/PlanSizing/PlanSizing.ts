import { Size } from '../../../Utils/Geometry'
import { noCase } from '../../../Utils/Utils'
import { PlanChartSizing } from '../PlanChart/PlanChart'
import { PlanHelpSizing } from '../PlanHelp'
import { PlanInputSizing } from '../PlanInput/PlanInput'
import { PlanSummaySizing } from '../PlanSummary/PlanSummary'
import { planSizingDesktop } from './PlanSizingDesktop'
import { planSizingLaptop } from './PlanSizingLaptop'
import { planSizingMobile } from './PlanSizingMobile'

export type PlanSizing = {
  input: PlanInputSizing
  help: PlanHelpSizing
  chart: PlanChartSizing
  summary: PlanSummaySizing
}

export function planSizing(
  layout: 'mobile' | 'laptop' | 'desktop',
  windowSize: Size,
  isSWR: boolean,
) {
  switch (layout) {
    case 'laptop':
      return planSizingLaptop(windowSize, isSWR)
    case 'desktop':
      return planSizingDesktop(windowSize, isSWR)
    case 'mobile':
      return planSizingMobile(windowSize, isSWR)
    default:
      noCase(layout)
  }
}
