import _ from 'lodash'
import React, {CSSProperties, useCallback, useEffect, useMemo} from 'react'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {fGet} from '../../../Utils/Utils'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {chartDrawDataLines} from '../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartMinMaxYAxis} from '../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {ChartPointer} from '../../Common/Chart/ChartComponent/ChartPointer'
import {ChartXAxis} from '../../Common/Chart/ChartComponent/ChartXAxis'
import {ChartReact, ChartReactState} from '../../Common/Chart/ChartReact'
import {ChartUtils} from '../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartData} from './TPAWChartData'

export type TPAWChartState = {
  data: TPAWChartData
  yRange: SimpleRange
  externalTopPadding: number
  animation: ChartAnimation|null
}

export const TPAWChart = React.memo(
  ({
    className = '',
    style,
    yAxisFormat,
    lastAgeIsLegacy,
    state: stateIn,
    animationForBoundsChange,
  }: {
    className?: string
    style?: CSSProperties
    state: TPAWChartState
    yAxisFormat: (x: number) => string
    animationForBoundsChange: ChartAnimation
    lastAgeIsLegacy: boolean
  }) => {
    const state = useMemo((): ChartReactState<TPAWChartData> => {
      const {yRange, data, externalTopPadding, animation} = stateIn
      const xyRange = {
        x: {start: data.age.start, end: data.age.end},
        y: yRange,
      }
      const padding = ({
        width,
        height: heightIn,
      }: {
        width: number
        height: number
      }) => {
        const baseTop = 30 + externalTopPadding
        const aspectIn = width / (heightIn - baseTop)
        const aspect =
          aspectIn < 2 ? linearFnFomPoints(1, 1.7, 2, 2)(aspectIn) : aspectIn
        const height = width / aspect

        return {
          left: 15,
          top: heightIn - height,
          bottom: 35,
          right: 15,
        }
      }
      return {xyRange, data, padding, animation}
    }, [stateIn])

    const pointerFormatX = useCallback(
      (data: TPAWChartData, x: number) =>
        x === data.age.end && lastAgeIsLegacy ? 'Legacy' : `Age ${x}`,
      [lastAgeIsLegacy]
    )

    const components = useMemo(
      () => {
        const minorLine = chartDrawDataLines<TPAWChartData>({
          lineWidth: 0.5,
          strokeStyle: ChartUtils.color.gray[400],
          dataFn: (data: TPAWChartData) => ({
            lines: data.percentiles
              .filter(x => !x.isHighlighted)
              .map(x => x.data),
            isXInGroup: data.isAgeInGroup,
          }),
        })
        const majorLine = chartDrawDataLines<TPAWChartData>({
          lineWidth: 1.2,
          strokeStyle: ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartData) => ({
            lines: data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => x.data),
            isXInGroup: data.isAgeInGroup,
          }),
        })

        const minMaxYAxis = new ChartMinMaxYAxis<TPAWChartData>(
          yAxisFormat,
          ChartUtils.color.gray[800],
          data => data.max.x,
          (data, x) => ({
            min: data.percentiles[0].data(x),
            max: fGet(_.last(data.percentiles)).data(x),
          })
        )
        const xAxis = new ChartXAxis<TPAWChartData>(
          (data, x) => (x === data.modelAgeEnd + 1 ? 'L' : `${x}`),
          data => data.age.retirement
        )

        const pointer = new ChartPointer<TPAWChartData>(
          data =>
            data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => ({line: x.data, label: `${x.percentile}`})),
          pointerFormatX,
          yAxisFormat,
          [xAxis]
        )
        const byName = {minorLine, majorLine, minMaxYAxis, xAxis, pointer}
        const arr = [minorLine, majorLine, minMaxYAxis, pointer]
        return {byName, arr}
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      []
    )

    useEffect(() => {
      components.byName.minMaxYAxis.format = yAxisFormat
      components.byName.pointer.formatY = yAxisFormat
      components.byName.pointer.formatX = pointerFormatX
    }, [yAxisFormat, components, pointerFormatX])

    return (
      <ChartReact<TPAWChartData>
        className={`${className}`}
        style={style}
        state={state}
        animationForBoundsChange={animationForBoundsChange}
        components={components.arr}
      />
    )
  }
)
