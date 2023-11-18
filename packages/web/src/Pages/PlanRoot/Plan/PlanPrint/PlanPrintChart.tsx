import React from 'react'
import {
  ChartYAxis,
  getYAxisGridInfo,
} from '../../../Common/Chart/ChartComponent/ChartYAxis'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { PlanResultsChartCustomComponenet } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartCustomComponent'
import { getPlanResultsChartBreakdown } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartBreakdown'
import { getPlanResultsChartRange } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartRange'
import { getPlanResultsChartXAxis } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartXAxis'
import { PlanResultsChartData } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartData'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  isPlanResultsChartSpendingTotalFundingSourcesType,
} from '../PlanResults/PlanResultsChartType'
import { noCase } from '@tpaw/common'

export const PlanPrintChart = React.memo(
  React.forwardRef(
    (
      {
        starting,
      }: {
        starting: {
          data: PlanResultsChartData
          sizing: ChartReactSizing
        }
      },
      ref: React.ForwardedRef<
        ChartReactStatefull<{ data: PlanResultsChartData }>
      >,
    ) => {
      return (
        <ChartReact<{ data: PlanResultsChartData }>
          ref={ref}
          starting={{
            data: { data: starting.data },
            sizing: starting.sizing,
            propsFn: ({ data }) => {
              return {
                dataRange: {
                  y: {
                    start: data.displayRange.y.start,
                    end: _getYAxisGridInfo(data).displayMax,
                  },
                  x: data.displayRange.x,
                },
                includeWidthOfLastX: false,
              }
            },
          }}
          components={components}
        />
      )
    },
  ),
)

const components = () => {
  const yAxis = new ChartYAxis<{ data: PlanResultsChartData }>(
    ({ params: { data } }) => {
      const { planColors } = data
      if (data.chartType === 'spending-total') {
        _getYAxisGridInfo(data)
      }
      return {
        format: data.formatY,
        gridInfo: _getYAxisGridInfo(data),
        style: {
          colors: {
            text: planColors.dark.rgb,
            line: planColors.shades.light[10].rgb,
          },
        },
      }
    },
  )
  return [
    getPlanResultsChartRange('print'),
    getPlanResultsChartBreakdown(),
    yAxis,
    new PlanResultsChartCustomComponenet(planPrintChartOutsidePlotAreaSizing.below),
    getPlanResultsChartXAxis(
      'person1',
      planPrintChartOutsidePlotAreaSizing.below,
    ),
    getPlanResultsChartXAxis(
      'person2',
      planPrintChartOutsidePlotAreaSizing.below,
    ),
  ]
}

export const planPrintChartOutsidePlotAreaSizing = {
  below: {
    gapToLine: 4,
    lineWidth: 1.5,
    xAxis: { height: 23, vertGapBetweenPeople: 2 },
  },
}

const _getYAxisGridInfo = (data: PlanResultsChartData) => {
  return getYAxisGridInfo({
    max: data.displayRange.y.end,
    maxGridLine: _getIsPercentage(data.chartType) ? 1 : null,
  })
}

const _getIsPercentage = (chartType: PlanResultsChartType) => {
  switch (chartType) {
    case 'asset-allocation-savings-portfolio':
    case 'asset-allocation-total-portfolio':
    case 'withdrawal':
      return true
    case 'portfolio':
    case 'spending-general':
    case 'spending-total':
      return false
    default:
      if (
        isPlanResultsChartSpendingTotalFundingSourcesType(chartType) ||
        isPlanResultsChartSpendingEssentialType(chartType) ||
        isPlanResultsChartSpendingDiscretionaryType(chartType)
      )
        return false
      noCase(chartType)
  }
}
