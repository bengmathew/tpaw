import {Size} from '../../../Utils/Geometry'
import {noCase} from '../../../Utils/Utils'
import {ChartPanelSizing} from '../ChartPanel/ChartPanel'
import {GuidePanelSizing} from '../GuidePanel'
import {ParamsInputSizing} from '../ParamsInput/ParamsInput'
import {ParamsInputSummaySizing} from '../ParamsInput/ParamsInputSummary/ParamsInputSummary2'
import {PlanHeadingSizing} from '../PlanHeading'
import {planSizingDesktop} from './PlanSizingDesktop'
import {planSizingLaptop} from './PlanSizingLaptop'
import {planSizingMobile} from './PlanSizingMobile'

export type PlanSizing = {
  heading: PlanHeadingSizing
  inputSummary: ParamsInputSummaySizing
  input: ParamsInputSizing
  guide: GuidePanelSizing
  chart: ChartPanelSizing
}

export function planSizing(
  layout: 'mobile' | 'laptop' | 'desktop',
  windowSize: Size
) {
  switch (layout) {
    case 'laptop':
      return planSizingLaptop(windowSize)
    case 'desktop':
      return planSizingDesktop(windowSize)
    case 'mobile':
      return planSizingMobile(windowSize)
    default:
      noCase(layout)
  }
}
