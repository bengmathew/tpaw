import { add } from 'lodash'
import { Size } from '../../../../Utils/Geometry'
import { noCase } from '../../../../Utils/Utils'
import { PlanChartPointerSizing } from '../PlanChartPointer/PlanChartPointer'
import { PlanContactSizing } from '../PlanContact/PlanContact'
import { PlanHelpSizing } from '../PlanHelp'
import { PlanInputSizing } from '../PlanInput/PlanInput'
import { PlanMenuSizing } from '../PlanMenu/PlanMenu'
import { PlanResultsSizing } from '../PlanResults/PlanResults'
import { PlanSummaySizing } from '../PlanSummary/PlanSummary'
import { planSizingDesktop } from './PlanSizingDesktop'
import { planSizingLaptop } from './PlanSizingLaptop'
import { planSizingMobile } from './PlanSizingMobile'

export type PlanSizing = {
  args: {
    layout: 'mobile' | 'laptop' | 'desktop'
    windowSize: Size
    scrollbarWidth: number
    isSWR: boolean
    tallPlanMenu: boolean
  }
  input: PlanInputSizing
  help: PlanHelpSizing
  chart: PlanResultsSizing
  summary: PlanSummaySizing
  menu: PlanMenuSizing
  contact: PlanContactSizing
  pointer: PlanChartPointerSizing
}

export function planSizing(
  layout: 'mobile' | 'laptop' | 'desktop',
  windowSize: Size,
  scrollbarWidth: number,
  isSWR: boolean,
  tallPlanMenu: boolean,
) {
  const addArgs = (x: Omit<PlanSizing, 'args'>): PlanSizing => ({
    ...x,
    args: { layout, windowSize, scrollbarWidth, isSWR, tallPlanMenu },
  })
  switch (layout) {
    case 'laptop':
      return addArgs(
        planSizingLaptop(windowSize, scrollbarWidth, isSWR, tallPlanMenu),
      )
    case 'desktop':
      return addArgs(
        planSizingDesktop(windowSize, scrollbarWidth, isSWR, tallPlanMenu),
      )
    case 'mobile':
      return addArgs(
        planSizingMobile(windowSize, scrollbarWidth, isSWR, tallPlanMenu),
      )
    default:
      noCase(layout)
  }
}
