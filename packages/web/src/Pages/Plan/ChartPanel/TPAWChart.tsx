import _ from 'lodash'
import React, { useCallback, useEffect, useMemo } from 'react'
import { RectExt, rectExt } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { fGet } from '../../../Utils/Utils'
import { ChartAnimation } from '../../Common/Chart/Chart'
import { chartDrawDataLines } from '../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { ChartMinMaxYAxis } from '../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import { ChartPointer } from '../../Common/Chart/ChartComponent/ChartPointer'
import { ChartXAxis } from '../../Common/Chart/ChartComponent/ChartXAxis'
import { ChartReact } from '../../Common/Chart/ChartReact'
import { ChartReactState } from '../../Common/Chart/ChartReactSub'
import { ChartUtils } from '../../Common/Chart/ChartUtils/ChartUtils'
import { chartDrawLegacy } from './LegacyComponents/ChartDrawLegacy'
import { TPAWChartData } from './TPAWChartData'
import { TPAWChartLegacyData } from './TPAWChartLegacyData'

export type TPAWChartState = {
  main: {data: TPAWChartData; yRange: SimpleRange}
  legacy: {data: TPAWChartLegacyData; yRange: SimpleRange; show: boolean}
  animation: ChartAnimation | null
}

export const TPAWChart = React.memo(
  ({
    yAxisFormat,
    state: stateIn,
    startingPosition,
    chartRef,
    extraPaddingTop
  }: {
    startingPosition: RectExt
    state: TPAWChartState
    yAxisFormat: (x: number) => string
    chartRef: (chart: {setPosition: (position: RectExt) => void}) => void
    extraPaddingTop:number
  }) => {
    const state = useMemo((): {
      main: ChartReactState<TPAWChartData>
      legacy: ChartReactState<TPAWChartLegacyData>
    } => {
      const {main: mainIn, legacy: legacyIn, animation} = stateIn
      const main = (() => {
        const xyRange = {
          x: mainIn.data.years.displayRange,
          y: mainIn.yRange,
        }
        const area = ({
          width,
          height: heightIn,
        }: {
          width: number
          height: number
        }) => {
          const baseTop = 35 + extraPaddingTop
          const aspectIn = width / (heightIn - baseTop)
          const aspect = Math.max(1.25, aspectIn)
          const height = width / aspect
          const viewport = rectExt({
            x: 0,
            y: 0,
            width: width - (legacyIn.show ? 115 : 0),
            height: heightIn,
          })
          const padding = {
            left: 15,
            top: heightIn - height,
            bottom: 35,
            right: 15,
          }
          return {viewport, padding}
        }
        return {area, xyRange, data: mainIn.data, alpha: 1, animation}
      })()

      const legacy = (() => {
        const xyRange = {
          x: {start: legacyIn.data.year, end: legacyIn.data.year + 0.1},
          y: legacyIn.yRange,
        }
        const area = ({width, height}: {width: number; height: number}) => {
          const mainArea = main.area({width, height})
          const overlap = width < 640 ? 0 : 0
          const viewport = rectExt({
            x: mainArea.viewport.right - overlap,
            y: 0,
            width: Math.max(25, width - mainArea.viewport.width) + overlap,
            bottom: mainArea.viewport.bottom,
          })
          const padding = {
            left: 0,
            top: mainArea.padding.top,
            // bottom: mainArea.padding.bottom,
            bottom: 17,
            right: 15,
          }
          return {viewport, padding}
        }
        return {
          area,
          xyRange,
          data: legacyIn.data,
          alpha: legacyIn.show ? 1 : 0,
          animation,
        }
      })()

      return {main, legacy}
    }, [extraPaddingTop, stateIn])

    const pointerFormatX = useCallback(
      (data: TPAWChartData, x: number) =>
        x === data.years.max + 1 ? 'Legacy' : `Age ${data.years.display(x)}`,
      []
    )

    const mainComponents = useMemo(
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
          (data, x) => data.years.display(x),
          (data, x) =>
            x === data.years.display(data.years.max) + 1 ? 'L' : `${x}`,
          data => data.years.retirement
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
        const byName = {minorLine, majorLine, minMaxYAxis, pointer, xAxis}
        const arr = [minorLine, majorLine, minMaxYAxis, pointer]
        return {byName, arr}
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      []
    )

    useEffect(() => {
      mainComponents.byName.minMaxYAxis.format = yAxisFormat
      mainComponents.byName.pointer.formatY = yAxisFormat
      mainComponents.byName.pointer.formatX = pointerFormatX
    }, [yAxisFormat, mainComponents, pointerFormatX])

    const legacyComponents = useMemo(() => {
      const legacy = chartDrawLegacy()
      return {
        arr: [legacy],
        byName: {legacy},
      }
    }, [])

    return (
      <ChartReact<[TPAWChartLegacyData, TPAWChartData]>
        charts={[
          {
            state: state.legacy,
            components: legacyComponents.arr,
            key: 'legacy',
            order: 0,
          },
          {
            state: state.main,
            components: mainComponents.arr,
            key: 'main',
            order: 1,
          },
        ]}
        startingPosition={startingPosition}
        chartRef={chartRef}
      />
    )
  }
)
