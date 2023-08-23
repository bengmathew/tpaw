import { ta } from 'date-fns/locale'
import { Size } from '../../../../Utils/Geometry'
import { noCase } from '../../../../Utils/Utils'
import { PlanResultsSizing } from '../PlanResults/PlanResults'
import { PlanHelpSizing } from '../PlanHelp'
import { PlanInputSizing } from '../PlanInput/PlanInput'
import { PlanMenuSizing } from '../PlanMenu/PlanMenu'
import { PlanSummaySizing } from '../PlanSummary/PlanSummary'
import { planSizingDesktop } from './PlanSizingDesktop'
import { planSizingLaptop } from './PlanSizingLaptop'
import { planSizingMobile } from './PlanSizingMobile'

export type PlanSizing = {
  input: PlanInputSizing
  help: PlanHelpSizing
  chart: PlanResultsSizing
  summary: PlanSummaySizing
  menu: PlanMenuSizing
}

export function planSizing(
  layout: 'mobile' | 'laptop' | 'desktop',
  windowSize: Size,
  scrollbarWidth: number,
  isSWR: boolean,
  tallPlanMenu: boolean,
) {
  switch (layout) {
    case 'laptop':
      return planSizingLaptop(windowSize, scrollbarWidth, isSWR, tallPlanMenu)
    case 'desktop':
      return planSizingDesktop(windowSize, scrollbarWidth, isSWR, tallPlanMenu)
    case 'mobile':
      return planSizingMobile(windowSize, scrollbarWidth, isSWR, tallPlanMenu)
    default:
      noCase(layout)
  }
}
