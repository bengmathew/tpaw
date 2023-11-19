import { block, fGet } from '@tpaw/common'
import { SimpleRange } from '../../../../../../Utils/SimpleRange'
import { ChartRange } from '../../../../../Common/Chart/ChartComponent/ChartRange'
import { ChartSalientPoints } from '../../../../../Common/Chart/ChartComponent/ChartSalientPoints'
import { PlanResultsChartData } from './PlanResultsChartData'

export const getPlanResultsChartSalientPoints = (
  rangeIds: ChartRange<unknown>['ids'],
) =>
  new ChartSalientPoints<{ data: PlanResultsChartData }>(({ data }) => {
    const { results: chartColors } = data.planColors
    return {
      salientPoints:
        data.type === 'range'
          ? block(() => {
              const { asMFN, withdrawalStartMonth } = data.planParamsExt
              const getPointsAtX = (
                xId: string,
                dataX: number,
                yLocations: ('start' | 'mid' | 'end')[],
              ) => {
                const dataY = fGet(data.range.yRangeByX[dataX])
                return yLocations.map((yLocation) => ({
                  id: `${xId}-Y${yLocation}`,
                  dataX,
                  dataY: dataY[yLocation],
                  label: data.formatY(dataY[yLocation]),
                  opacity: 1,
                  align:
                    yLocation === 'start'
                      ? ('below' as const)
                      : ('above' as const),
                  collisionDataId: rangeIds[yLocation],
                  overlapDetectionIsVertOnly: false,
                }))
              }
              const unfiltered = [
                ...getPointsAtX('rangeXStart', data.displayRange.x.start, [
                  'end',
                  'mid',
                  'start',
                ]),
                ...getPointsAtX('rangeXEnd', data.displayRange.x.end, [
                  'end',
                  'mid',
                  'start',
                ]),
              ]
              ;(['min', 'max'] as const).forEach((minOrMax) => {
                const info = data.range.minMax[minOrMax]
                if (
                  (minOrMax === 'min' &&
                    info.y >
                      Math.min(...unfiltered.map((x) => x.dataY)) * 0.9) ||
                  (minOrMax === 'max' &&
                    info.y < Math.max(...unfiltered.map((x) => x.dataY)) * 1.1)
                ) {
                  return
                }
                unfiltered.push({
                  id: `range${minOrMax}`,
                  dataX: info.x,
                  dataY: info.y,
                  label: data.formatY(info.y),
                  opacity: 1,
                  align:
                    minOrMax === 'min'
                      ? ('below' as const)
                      : ('above' as const),
                  collisionDataId:
                    rangeIds[minOrMax === 'min' ? 'start' : 'end'],
                  overlapDetectionIsVertOnly: true,
                })
              })
              return unfiltered.map((x, i) =>
                unfiltered
                  .slice(0, i)
                  .some(
                    (prev) => x.dataX === prev.dataX && x.label === prev.label,
                  )
                  ? { ...x, opacity: 0 }
                  : x,
              )
            })
          : block(() => {
              const getTotalPointAtX = (id: string, dataX: number) => {
                const dataY = fGet(data.breakdown.total.yByX[dataX])
                return {
                  id,
                  dataX,
                  dataY,
                  label: data.formatY(dataY),
                  opacity: 1,
                  align: 'above' as const,
                  collisionDataId: `breakdown-total`,
                }
              }
              return [
                getTotalPointAtX('breakdown-start', data.displayRange.x.start),
                getTotalPointAtX('breakdown-end', data.displayRange.x.end),
                // ...(['min', 'max'] as const).map((minOrMax) => ({
                //   id: `breakdown-${minOrMax}`,
                //   dataX: data.breakdown.total.minMax[minOrMax].x,
                //   dataY: data.breakdown.total.minMax[minOrMax].y,
                //   label: data.formatY(
                //     data.breakdown.total.minMax[minOrMax].y,
                //   ),
                //   opacity: 1,
                //   align: 'above' as const,
                //   collisionDataId: `breakdown-total`,
                // })),
              ]
            }),
      colors: chartColors,
      fontSize: 11,
    }
  })
