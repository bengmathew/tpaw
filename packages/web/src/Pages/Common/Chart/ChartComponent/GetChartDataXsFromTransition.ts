import { letIn } from '@tpaw/common'
import _ from 'lodash'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { Transition, transitionTransform } from '../../../../Utils/Transition'
import { ChartDataRange, ChartStateDerived } from '../Chart'

// export const getChartDataXsFromTransition = (
//   transition: Transition<{ derivedState: ChartStateDerived }>,
// ) =>
//   letIn(_getChartDataXRangeFromTransition(transition), ({ start, end }) =>
//     _.range(start, end),
//   )

//  const _getChartDataXRangeFromTransition = (
//   transition: Transition<{ derivedState: ChartStateDerived }>,
// ): SimpleRange =>
//   letIn(
//     transitionTransform(
//       transition,
//       ({ derivedState: { scale, plotArea } }) => ({
//         start: Math.ceil(scale.x.inverse(plotArea.x)),
//         end: Math.floor(scale.x.inverse(plotArea.x + plotArea.width)) + 1,
//       }),
//     ),
//     ({ from, target, progress }) =>
//       progress === 1
//         ? target
//         : {
//             start: Math.min(from.start, target.start),
//             end: Math.max(from.end, target.end),
//           },
//   )

export const getChartDataXsFromTransition = (
  transition: Transition<{ dataRange: ChartDataRange }>,
) => {
  const from = transition.from.dataRange.x
  const target = transition.from.dataRange.x
  const start = Math.min(from.start, target.start)
  const end = Math.max(from.end, target.end)
  return _.range(start, end + 1)
}
