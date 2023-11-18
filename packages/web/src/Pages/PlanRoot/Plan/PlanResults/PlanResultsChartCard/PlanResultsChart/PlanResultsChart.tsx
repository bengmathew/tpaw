import React, { useMemo } from 'react'
import { SimpleRange } from '../../../../../../Utils/SimpleRange'
import { ChartComponent } from '../../../../../Common/Chart/ChartComponent/ChartComponent'
import { ChartPointerPortal } from '../../../../../Common/Chart/ChartComponent/ChartPointerPortal'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../../../Common/Chart/ChartReact'
import { PlanResultsChartCustomComponenet } from './PlanResultsChartCustomComponent'
import { getPlanResultsChartBreakdown } from './GetPlanResultsChartBreakdown'
import { getPlanResultsChartPointer } from './GetPlanResultsChartPointer'
import { getPlanResultsChartRange } from './GetPlanResultsChartRange'
import { getPlanResultsChartSalientPoints } from './GetPlanResultsChartSalientPoints'
import { getPlanResultsChartXAxis } from './GetPlanResultsChartXAxis'
import { PlanResultsChartData } from './PlanResultsChartData'

type _ChartData = { data: PlanResultsChartData; yRange: SimpleRange }
export const PlanResultsChart = React.memo(
  React.forwardRef(
    (
      {
        starting,
        chartPointerPortal,
        onHover,
        capturePointerEvents,
      }: {
        starting: {
          data: _ChartData
          sizing: ChartReactSizing
        }
        chartPointerPortal: ChartPointerPortal
        onHover: (hover: boolean) => void
        capturePointerEvents: boolean
      },
      ref: React.ForwardedRef<ChartReactStatefull<_ChartData>>,
    ) => {
      const components = useMemo(
        () => getComponets(chartPointerPortal),
        [chartPointerPortal],
      )
      return (
        <ChartReact<_ChartData>
          ref={ref}
          starting={{
            ...starting,
            propsFn: ({ data, yRange }) => ({
              dataRange: { y: yRange, x: data.displayRange.x },
              includeWidthOfLastX: false,
            }),
            debug: true,
          }}
          components={components}
          onHover={onHover}
          captureTouchEvents={capturePointerEvents}
        />
      )
    },
  ),
)

const getComponets =
  (chartPointerPortal: ChartPointerPortal) =>
  (): ChartComponent<_ChartData>[] => {
    const range = getPlanResultsChartRange('main')
    const breakdown = getPlanResultsChartBreakdown()
    return [
      range,
      breakdown,
      new PlanResultsChartCustomComponenet(planResultChartOutsidePlotAreaSizing.below),
      getPlanResultsChartXAxis(
        'person1',
        planResultChartOutsidePlotAreaSizing.below,
      ),
      getPlanResultsChartXAxis(
        'person2',
        planResultChartOutsidePlotAreaSizing.below,
      ),
      getPlanResultsChartSalientPoints(range.ids),
      getPlanResultsChartPointer(chartPointerPortal, range.ids, breakdown.ids),
    ]
  }

export const planResultChartOutsidePlotAreaSizing = {
  below: {
    gapToLine: 5,
    lineWidth: 1.5,
    xAxis: { height: 23, vertGapBetweenPeople: 2 },
  },
}
