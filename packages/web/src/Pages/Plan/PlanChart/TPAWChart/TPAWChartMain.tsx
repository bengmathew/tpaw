import { linearFnFomPoints } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { newPadding } from '../../../../Utils/Geometry'
import { fGet, noCase } from '../../../../Utils/Utils'
import { ChartXYRange } from '../../../Common/Chart/Chart'
import { chartDrawDataLines } from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
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
  const minorLine = chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 0.5,
    strokeStyle: ChartUtils.color.gray[400],
    dataFn: (data: TPAWChartDataMain) =>
      data.series.type === 'percentiles'
        ? {
            lines: data.series.percentiles
              .filter((x) => !x.isHighlighted)
              .map((x) => x.data),
          }
        : data.series.type === 'labeledLines'
        ? {
            lines: _.times(
              data.series.percentiles.length -
                data.series.highlightedPercentiles.length,
            ).map(() => null),
          }
        : noCase(data.series),
  })
  const majorLine = chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2,
    strokeStyle: ChartUtils.color.gray[500],
    dataFn: (data: TPAWChartDataMain) =>
      data.series.type === 'percentiles'
        ? {
            lines: data.series.percentiles
              .filter((x) => x.isHighlighted)
              .map((x) => x.data),
          }
        : {
            lines: [
              data.series.lines[0].data,
              ..._.times(data.series.highlightedPercentiles.length - 3).map(
                () => null,
              ),
              data.series.lines[1].data,
              data.series.lines[2].data,
            ],
          },
  })

  const minMaxYAxis = new ChartMinMaxYAxis<TPAWChartDataMain>(
    (data, x) => data.yFormat(x),
    ChartUtils.color.gray[800],
    (data) => data.max.x,
    (data, x) => {
      switch (data.series.type) {
        case 'percentiles':
          return {
            min: data.series.percentiles[0].data(x),
            max: fGet(_.last(data.series.percentiles)).data(x),
          }
        case 'labeledLines':
          const ys = data.series.lines.map(({ data }) => data(x))
          return {
            min: Math.min(...ys),
            max: Math.max(...ys),
          }
        default:
          noCase(data.series)
      }
    },
  )

  const pointer = hidePointer
    ? null
    : new ChartPointer<TPAWChartDataMain>(
        (data) => {
          switch (data.series.type) {
            case 'percentiles':
              return data.series.percentiles
                .filter((x) => x.isHighlighted)
                .map((x) => ({ line: x.data, label: `${x.percentile}` }))
            case 'labeledLines': {
              const tpaw = {
                line: data.series.lines[0].data,
                label: data.series.lines[0].label,
              }
              const spaw = {
                line: data.series.lines[1].data,
                label: data.series.lines[1].label,
              }
              const swr = {
                line: data.series.lines[2].data,
                label: data.series.lines[2].label,
              }
              const ordered = _.sortBy([tpaw, spaw, swr], (x) =>
                x.line(data.years.displayRange.start),
              )

              return [
                ordered[0],
                ordered[1],
                ..._.times(
                  data.series.highlightedPercentiles.length - ordered.length,
                ).map(() => null),
                ordered[2],
              ]
            }
            default:
              noCase(data.series)
          }
        },
        ({ dataTransition, derivedState }) => {
          const data = dataTransition.target
          const { viewport } = derivedState.curr
          const { pickPerson, asYFN, years } = data.paramsExt
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
                pickPerson(person).ages.current + dataX
              return dataX === data.years.max + 1
                ? 'Legacy'
                : data.params.people.withPartner
                ? `Ages ${
                    dataX > asYFN(years.person1.max) ? '＿' : ageX('person1')
                  },${
                    dataX > asYFN(years.person2.max) ? '＿' : ageX('person2')
                  }`
                : `Age ${ageX('person1')}`
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
      const { params, pickPerson, years, asYFN } =
        dataTransition.target.paramsExt
      if (person === 'person2' && !params.people.withPartner) {
        // We are invisible anyway, so it does not matter, but this will preven
        // asserts. Hacky, I know.
        person = 'person1'
      }
      const dataXTransform = (dataX: number) =>
        dataX + pickPerson(person).ages.current
      const sizing = tpawChartMainXAxisSizing(viewport.width)
      const dyn = _dynSizing(viewport.width)

      return {
        type: (x) =>
          x % 10 === 0 ? 'large' : x % 5 === 0 ? 'medium' : 'small',
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
            : result(dyn(1, 2), color['600'], ChartUtils.getFont(dyn(8, 9)))
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
        formatLabel: (transformedDataX) => `${transformedDataX}`,
        shouldLabel: (pixelsPerTick, type) => {
          switch (type) {
            case 'small':
              return false
            case 'medium':
              return pixelsPerTick > 15
            case 'large':
              return true
          }
        },
        visible: person === 'person1' ? true : params.people.withPartner,
        yOffset:
          4 +
          (params.people.withPartner
            ? +sizing.gap +
              (person === 'person1' ? 0 : sizing.height + sizing.gap)
            : sizing.gap),
        maxDataX: asYFN(years[person].max),
        retirementDataX: asYFN(years[person].retirement),
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
    minorLine,
    majorLine,
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
