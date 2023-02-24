import { linearFnFomPoints, noCase } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { newPadding } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { ChartXYRange } from '../../../Common/Chart/Chart'
import { chartDrawDataLines } from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { chartDrawDataRangeBand } from '../../../Common/Chart/ChartComponent/ChartDrawRangeBand'
import { ChartMinMaxYAxis } from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import { ChartPointer } from '../../../Common/Chart/ChartComponent/ChartPointer'
import { ChartXAxis } from '../../../Common/Chart/ChartComponent/ChartXAxis'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { ChartDrawMain } from '../CustomComponents/ChartDrawMain'
import { TPAWChartDataMain } from './TPAWChartDataMain'

export const TPAWChartMain = React.memo(
  React.forwardRef(
    (
      {
        starting,
        hidePointer = false,
      }: {
        starting: {
          data: TPAWChartDataMain
          xyRange: ChartXYRange
          sizing: ChartReactSizing
        }
        hidePointer?: boolean
      },
      ref: React.ForwardedRef<ChartReactStatefull<TPAWChartDataMain>>,
    ) => {
      return (
        <ChartReact<TPAWChartDataMain>
          ref={ref}
          starting={starting}
          components={components(hidePointer)}
        />
      )
    },
  ),
)

const components = (hidePointer: boolean) => () => {
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

  const minMaxYAxis = new ChartMinMaxYAxis<TPAWChartDataMain>(
    (data, x) => data.yFormat(x),
    ChartUtils.color.gray[800],
    (data) => data.max.x,
    (data, x) => ({
      min: data.percentiles[0].data(x),
      max: fGet(_.last(data.percentiles)).data(x),
    }),
  )

  const pointer = hidePointer
    ? null
    : new ChartPointer<TPAWChartDataMain>(
        (data) => {
          return data.percentiles.map((x) => ({
            line: x.data,
            label: `${x.percentile}`,
          }))
        },
        ({ dataTransition, derivedState }) => {
          const data = dataTransition.target
          const { viewport } = derivedState.curr
          const { pickPerson, asMFN, months } = data.paramsExt
          const scaled = (at200: number, at500: number) =>
            _.clamp(
              linearFnFomPoints(200, at200, 500, at500)(viewport.width),
              at200,
              at500,
            )
          return {
            subHeading: 'Percentiles',
            formatX: (dataX: number) => {
              const ageX = (person: 'person1' | 'person2') =>
                pickPerson(person).ages.currentMonth + dataX
              return dataX === data.months.max + 1
                ? { type: 'legacy' }
                : data.params.people.withPartner
                ? {
                    type: 'withPartner',
                    ageInMonths:
                      dataX > asMFN(months.person1.max)
                        ? null
                        : ageX('person1'),
                    partnerAgeInMonths:
                      dataX > asMFN(months.person2.max)
                        ? null
                        : ageX('person2'),
                  }
                : { type: 'withoutPartner', ageInMonths: ageX('person1') }
            },
            formatY: data.yFormat,
            showTh: true,
            pad: {
              vert: {
                top: scaled(8, 10),
                between: scaled(6, 9),
                bottom: scaled(8, 10),
              },
              horz: {
                edge: scaled(8, 10),
                between: scaled(5, 20),
                outside: {
                  lineLength: 35 * scaled(0.3, 1),
                  margin: scaled(0, 25),
                },
              },
            },
          }
        },
      )

  const colorCode = {
    person1: ChartUtils.color.theme1,
    person2: ChartUtils.color.theme1,
  }
  const xAxis = (personIn: 'person1' | 'person2') =>
    new ChartXAxis<TPAWChartDataMain>(({ dataTransition, derivedState }) => {
      let person = personIn
      const { viewport } = derivedState.curr
      const { params, pickPerson, months, asMFN } =
        dataTransition.target.paramsExt
      if (person === 'person2' && !params.people.withPartner) {
        // We are invisible anyway, so it does not matter, but this will prevent
        // asserts. Hacky, I know.
        person = 'person1'
      }
      const dataXTransform = (dataX: number) =>
        dataX + pickPerson(person).ages.currentMonth
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
        visible: person === 'person1' ? true : params.people.withPartner,
        yOffset:
          6 +
          (params.people.withPartner
            ? +sizing.gap +
              (person === 'person1' ? 0 : sizing.height + sizing.gap)
            : sizing.gap),
        maxDataX: asMFN(months[person].max),
        retirementDataX: asMFN(months[person].retirement),
        label: person === 'person1' ? 'Your Age' : `Partner's Age`,
        tempPerson: person,
        height: sizing.height,
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

  const custom = new ChartDrawMain()
  return _.compact([
    // minorLine,
    rangeBand,
    medianLine,
    minMaxYAxis,
    custom,
    xAxis('person1'),
    xAxis('person2'),
    pointer,
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
