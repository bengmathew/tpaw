import { linearFnFomPoints, noCase } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { newPadding } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { ChartXYRange } from '../../../Common/Chart/Chart'
import { chartDrawDataLines } from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { chartDrawDataRangeBand } from '../../../Common/Chart/ChartComponent/ChartDrawRangeBand'
import { ChartXAxis } from '../../../Common/Chart/ChartComponent/ChartXAxis'
import {
  ChartYAxis,
  getYAxisGridInfo,
} from '../../../Common/Chart/ChartComponent/ChartYAxis'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { ChartDrawMain } from '../CustomComponents/ChartDrawMain'
import { TPAWChartDataMain } from './TPAWChartDataMain'
import {
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
} from '../PlanChartType'

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
  const medianLine = chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2,
    strokeStyle: ChartUtils.color.orange[400],
    dataFn: (data: TPAWChartDataMain) => ({
      lines: data.percentiles
        .filter((x) => x.percentile === 50)
        .map((x) => x.data),
    }),
  })
  const rangeBand = chartDrawDataRangeBand<TPAWChartDataMain>({
    fillStyle: ChartUtils.color.orange[100],
    dataFn: (data: TPAWChartDataMain) => ({
      min: fGet(_.first(data.percentiles)).data,
      max: fGet(_.last(data.percentiles)).data,
    }),
  })

  const colorCode = {
    person1: ChartUtils.color.theme1,
    person2: ChartUtils.color.theme1,
  }
  const xAxis = (personIn: 'person1' | 'person2') =>
    new ChartXAxis<TPAWChartDataMain>(({ dataTransition, derivedState }) => {
      let person = personIn
      const { viewport } = derivedState.curr
      const { params, months, asMFN, getCurrentAgeOfPerson } =
        dataTransition.target.paramsExt
      if (person === 'person2' && !params.plan.people.withPartner) {
        // We are invisible anyway, so it does not matter, but this will prevent
        // asserts. Hacky, I know.
        person = 'person1'
      }
      const dataXTransform = (dataX: number) =>
        dataX + getCurrentAgeOfPerson(person).inMonths
      const sizing = tpawChartMainXAxisSizing(viewport.width)
      const dyn = _dynSizing(viewport.width)

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
          const color =
            person === 'person1' ? ChartUtils.color.gray : ChartUtils.color.gray
          return type === 'large'
            ? result(
                dyn(4, 6),
                color[900],
                ChartUtils.getFont(dyn(8, 10), '500'),
              )
            : type === 'medium'
            ? result(dyn(3, 4), color['600'], ChartUtils.getFont(dyn(8, 9)))
            : result(dyn(1, 2), color['500'], ChartUtils.getFont(dyn(8, 9)))
        },
        style: {
          background: {
            retired: ChartUtils.color.gray[300],
            notRetired: ChartUtils.color.gray[200],
          },
        },
        pointerStyle: {
          color: {
            fill: colorCode[person],
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
        visible: person === 'person1' ? true : params.plan.people.withPartner,
        yOffset:
          6 +
          (params.plan.people.withPartner
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
          color: ChartUtils.color.gray[700],
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
            isPlanChartSpendingDiscretionaryType(data.type) ||
            isPlanChartSpendingEssentialType(data.type)
          )
            return null
          noCase(data.type)
        })(),
      }),
    {
      text: ChartUtils.color.gray[800],
      line: ChartUtils.color.gray[300],
    },
  )

  const custom = new ChartDrawMain(ChartUtils.color.gray[700], 2)
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
