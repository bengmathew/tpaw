import {Size} from '../../../Utils/Geometry'
import {noCase} from '../../../Utils/Utils'
import {PlanInputSizing} from '../PlanInput/PlanInput'
import {PlanChartSizing} from '../PlanChart/PlanChart'
import {PlanResultsSizing} from '../PlanResults'
import {PlanSummaySizing} from '../PlanSummary/PlanSummary'
import {PlanWelcomeSizing} from '../PlanWelcome'
import {planSizingDesktop} from './PlanSizingDesktop'
import {planSizingLaptop} from './PlanSizingLaptop'
import {planSizingMobile} from './PlanSizingMobile'
import { SimulationInfo } from '../../App/WithSimulation'


export type PlanSizing = {
  welcome: PlanWelcomeSizing
  input: PlanInputSizing
  results: PlanResultsSizing
  chart: PlanChartSizing
  summary: PlanSummaySizing
}

export function planSizing(
  layout: 'mobile' | 'laptop' | 'desktop',
  windowSize: Size,
  isSWR:boolean
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
