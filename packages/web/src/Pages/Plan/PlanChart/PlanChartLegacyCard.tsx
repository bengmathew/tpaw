import React, {useEffect, useMemo, useRef, useState} from 'react'
import {cssEaseAsGSAPEase} from '../../../Utils/CSSEaseAsGSAPEase'
import {
  insetCSSStyle,
  newPadding,
  originCSSStyle,
  Padding,
  rectExt,
  InsetExt,
} from '../../../Utils/Geometry'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {useChartData} from '../../App/WithChartData'
import {ChartReactStatefull} from '../../Common/Chart/ChartReact'
import {
  PlanChartInternalTransitionState,
  planChartNormalAnimation,
} from './PlanChart'
import {TPAWChartDataLegacy} from './TPAWChart/TPAWChartDataLegacy'
import {TPAWChartLegacy} from './TPAWChart/TPAWChartLegacy'

type PlanChartLegacyCardSizing = {
  dynamic: Record<
    PlanChartInternalTransitionState,
    {
      legacyCard: {
        inset: InsetExt
        visibility: 'visible' | 'hidden'
      }
    }
  >
  fixed: {cardPadding: Padding}
}
export const PlanChartLegacyCard = React.memo(
  ({
    yRange,
    sizing,
    transition,
  }: {
    yRange: SimpleRange
    sizing: PlanChartLegacyCardSizing
    transition: {target: PlanChartInternalTransitionState; duration: number}
  }) => {
    const chartLegacyData = useChartData().legacy

    const chartRef = useRef<ChartReactStatefull<TPAWChartDataLegacy> | null>(
      null
    )
    const targetSizing = useMemo(
      () => _transformSizing(sizing.dynamic[transition.target]),
      [sizing.dynamic, transition.target]
    )
    useEffect(() => {
      chartRef.current?.setSizing(targetSizing.chart, {
        duration: transition.duration,
        ease: cssEaseAsGSAPEase,
      })
    }, [targetSizing, transition.duration])

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const data = chartLegacyData
      const prevState = chart.getState()
      chart.setState(data, prevState.xyRange, planChartNormalAnimation)
    }, [chartLegacyData])

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const prevState = chart.getState()
      chart.setState(
        prevState.data,
        {x: prevState.xyRange.x, y: yRange},
        planChartNormalAnimation
      )
    }, [yRange])
    const [hiddenAtTransitionEnd, setHiddenAtTransitionEnd] = useState(
      targetSizing.visibility === 'hidden'
    )
    const hide = targetSizing.visibility === 'hidden' && hiddenAtTransitionEnd

    return (
      <div
        className="absolute bg-cardBG rounded-2xl overflow-hidden"
        style={{
          // Needed to set visibility hidden because on Android when legacy
          // card was out of sight but visibility:visible performance dropped
          // significantly.
          visibility: hide ? 'hidden' : 'visible',
          transitionProperty: 'right, bottom, top, left, width, height',
          transitionDuration: `${transition.duration}ms`,
          // transitionTimingFunction:'linear',
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          right:`${targetSizing.inset.right}px`,
          top:`${targetSizing.inset.top}px`,
          width:`${targetSizing.inset.width}px`,
          height:`${targetSizing.inset.height}px`,
          // ...insetCSSStyle(targetSizing.inset)
        }}
        onTransitionEnd={() =>
          setHiddenAtTransitionEnd(targetSizing.visibility === 'hidden')
        }
      >
        <h2
          className="absolute font-bold text-[16px] sm:text-[22px] text-center"
          style={{
            transitionProperty: 'transform, width',
            transitionDuration: `${transition.duration}ms`,
            transform: `translate(0px, ${sizing.fixed.cardPadding.top}px)`,
            ...originCSSStyle({x: 0, y: 0}),
            width: `${targetSizing.inset.width}px`,
          }}
        >
          Legacy
        </h2>
        <TPAWChartLegacy
          starting={{
            data: chartLegacyData,
            xyRange: chartLegacyData.xyDisplayRange,
            sizing: targetSizing.chart,
          }}
          ref={chartRef}
        />
      </div>
    )
  }
)

const _transformSizing = ({
  legacyCard,
}: PlanChartLegacyCardSizing['dynamic']['hiddenWithLegacy']) => {
  const {inset} = legacyCard
  const chart = {
    position: rectExt({
      x: 0,
      y: 0,
      height: inset.height,
      width: inset.width,
    }),
    padding: newPadding({horz: 0, top: 60, bottom: 10}),
  }
  return {...legacyCard, chart}
}
