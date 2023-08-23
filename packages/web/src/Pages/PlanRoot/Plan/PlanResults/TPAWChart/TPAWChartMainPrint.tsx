import { fGet, linearFnFomPoints, noCase } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { newPadding } from '../../../../../Utils/Geometry'
import { ChartXYRange } from '../../../../Common/Chart/Chart'
import { chartDrawDataLines } from '../../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { chartDrawDataRangeBand } from '../../../../Common/Chart/ChartComponent/ChartDrawRangeBand'
import { ChartXAxis } from '../../../../Common/Chart/ChartComponent/ChartXAxis'
import {
  ChartYAxis,
  getYAxisGridInfo,
} from '../../../../Common/Chart/ChartComponent/ChartYAxis'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../../Common/Chart/ChartReact'
import { ChartUtils } from '../../../../Common/Chart/ChartUtils/ChartUtils'
import { colors, gray, mainPlanColors, orange } from '../../UsePlanColors'
import { ChartDrawMain } from '../CustomComponents/ChartDrawMain'
import {
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
} from '../PlanResultsChartType'
import { TPAWChartDataMain } from './TPAWChartDataMain'

export const TPAWChartMainPrint = React.memo(
  React.forwardRef(
    (
      {
        starting,
      }: {
        starting: {
          data: TPAWChartDataMain
          xyRange: ChartXYRange
          sizing: ChartReactSizing
        }
      },
      ref: React.ForwardedRef<ChartReactStatefull<TPAWChartDataMain>>,
    ) => {
      return (
        <ChartReact<TPAWChartDataMain>
          ref={ref}
          starting={starting}
          components={components()}
        />
      )
    },
  ),
)

const components = () => () => {
  const { results: chartColors } = mainPlanColors
  const medianLine = chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2,
    strokeStyle: chartColors.medianLine,
    dataFn: (data: TPAWChartDataMain) => ({
      lines: data.percentiles
        .filter((x) => x.percentile === 50)
        .map((x) => x.data),
    }),
  })
  const rangeBand = chartDrawDataRangeBand<TPAWChartDataMain>({
    fillStyle: chartColors.rangeBand,
    dataFn: (data: TPAWChartDataMain) => ({
      min: fGet(_.first(data.percentiles)).data,
      max: fGet(_.last(data.percentiles)).data,
    }),
  })

  const xAxis = (personIn: 'person1' | 'person2') =>
    new ChartXAxis<TPAWChartDataMain>(({ dataTransition, derivedState }) => {
      let person = personIn
      const { viewport } = derivedState.curr
      const { planParams, months, asMFN, getCurrentAgeOfPerson } =
        dataTransition.target.planParamsExt
      if (person === 'person2' && !planParams.people.withPartner) {
        // We are invisible anyway, so it does not matter, but this will prevent
        // asserts. Hacky, I know.
        person = 'person1'
      }
      const dataXTransform = (dataX: number) =>
        dataX + getCurrentAgeOfPerson(person).inMonths
      const sizing = tpawChartMainXAxisSizing(viewport.width)
      const dyn = _dynSizing(viewport.width)

      const colors = chartColors.xAxis
      return {
        type: (monthFromNow) => {
          if (monthFromNow % 12 !== 0) return 'none'
          const yearFromNow = Math.round(monthFromNow / 12)
          return yearFromNow % 10 === 0
            ? 'large'
            : yearFromNow % 5 === 0
            ? 'medium'
            : 'small'
        },
        tickStyle: (type) => {
          const result = (length: number, color: string, font: string) => ({
            length,
            color,
            font,
            gap: 2,
          })
          const color = person === 'person1' ? gray : gray
          return type === 'large'
            ? result(
                dyn(4, 6),
                colors.tick.large,
                ChartUtils.getFont(dyn(8, 10), '500'),
              )
            : type === 'medium'
            ? result(
                dyn(3, 4),
                colors.tick.medium,
                ChartUtils.getFont(dyn(8, 9)),
              )
            : result(
                dyn(1, 2),
                colors.tick.small,
                ChartUtils.getFont(dyn(8, 9)),
              )
        },
        style: {
          background: {
            retired: colors.bg.retired,
            notRetired: colors.bg.notRetired,
          },
        },
        pointerStyle: {
          color: {
            fill: colors.pointerBG,
          },
          height: dyn(6, 7),
        },
        padding: newPadding({ horz: 1, vert: 1 }),
        dataXTransform,
        formatLabel: (transformedDataX) => `${transformedDataX / 12}`,
        shouldLabel: (pixelsPerTick, type) => {
          const pixelsPerYear = pixelsPerTick * 12
          switch (type) {
            case 'small':
              return false
            case 'medium':
              return pixelsPerYear > 15
            case 'large':
              return true
            default:
              noCase(type)
          }
        },
        visible: person === 'person1' ? true : planParams.people.withPartner,
        yOffset:
          4 +
          (planParams.people.withPartner
            ? +sizing.gap +
              (person === 'person1' ? 0 : sizing.height + sizing.gap)
            : sizing.gap),
        maxDataX: asMFN(months[person].max),
        retirementDataX: asMFN(months[person].retirement),
        label: person === 'person1' ? 'Your Age' : `Partner's Age`,
        tempPerson: person,
        height: sizing.height,
        hidePointer: true,
        labelStyle: {
          font: ChartUtils.getFont(dyn(12, 14), '600'),
          color: gray[700],
          padding: {
            left: dyn(15, 20),
            bottom: dyn(2, 3),
          },
        },
      }
    })

  const yAxis = new ChartYAxis<TPAWChartDataMain>(
    (data, x) => data.yFormat(x),
    (data) =>
      getYAxisGridInfo({
        max: data.yDisplayRange.end,
        maxGridLine: (() => {
          if (
            data.type === 'asset-allocation-savings-portfolio' ||
            data.type === 'asset-allocation-total-portfolio' ||
            data.type === 'withdrawal'
          )
            return 1
          if (
            data.type === 'spending-total' ||
            data.type === 'spending-general' ||
            data.type === 'portfolio' ||
            isPlanResultsChartSpendingDiscretionaryType(data.type) ||
            isPlanResultsChartSpendingEssentialType(data.type)
          )
            return null
          noCase(data.type)
        })(),
      }),
    {
      text: gray[800],
      line: gray[400],
    },
  )

  const custom = new ChartDrawMain({
    xAxisLineColor: chartColors.xAxis.line,
    lineWidth: 1,
  })
  return _.compact([
    rangeBand,
    medianLine,
    custom,
    xAxis('person1'),
    xAxis('person2'),
    yAxis,
  ])
}

export const tpawChartMainXAxisSizing = (viewportWidth: number) => ({
  height: _dynSizing(viewportWidth)(22, 23),
  gap: 2,
})

const _dynSizing = (viewportWidth: number) => (at250: number, at500: number) =>
  _.clamp(
    linearFnFomPoints(250, at250, 500, at500)(viewportWidth),
    at250,
    at500,
  )
