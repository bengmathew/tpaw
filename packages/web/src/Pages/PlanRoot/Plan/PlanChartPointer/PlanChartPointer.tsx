import React, { useCallback, useEffect, useMemo, useRef } from 'react'
import { RectExt, newPadding, regionCSSStyle } from '../../../../Utils/Geometry'
import { ChartPointerPortal } from '../../../Common/Chart/ChartComponent/ChartPointerPortal'
import {
    ChartReact,
    ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { PlanTransitionState } from '../PlanTransition'

type _FixedSizing = {
  region: RectExt
  chartPosition: { bottom: number; left: number }
}

export type PlanChartPointerSizing = {
  fixed: {
    input: _FixedSizing
    summary: _FixedSizing
    help: _FixedSizing
  }
}

export const PlanChartPointer = React.memo(
  ({
    sizing,
    planTransition,
    chartPointerPortal,
  }: {
    sizing: PlanChartPointerSizing
    planTransition: { target: PlanTransitionState; duration: number }
    chartPointerPortal: ChartPointerPortal
  }) => {
    const isSummary = planTransition.target.section === 'summary'
    const isHelp = planTransition.target.section === 'help'
    const fixedSizing = useMemo(
      () =>
        isSummary
          ? sizing.fixed.summary
          : isHelp
          ? sizing.fixed.help
          : sizing.fixed.input,
      [
        isHelp,
        isSummary,
        sizing.fixed.help,
        sizing.fixed.input,
        sizing.fixed.summary,
      ],
    )

    const chartRef = useRef<ChartReactStatefull<null>>(null)
    useEffect(() => {
      if (!chartRef.current) return
      chartRef.current.setSizing(
        {
          position: { ...fixedSizing.region, x: 0, y: 0 },
          padding: newPadding(0),
        },
        null,
      )
    }, [fixedSizing])

    const components = useCallback(
      () => [chartPointerPortal],
      [chartPointerPortal],
    )

    return (
      <div
        className="absolute overflow-hidden z-[60] pointer-events-none "
        style={{ ...regionCSSStyle(fixedSizing.region) }}
      >
        <ChartReact
          ref={chartRef}
          starting={{
            data: null,
            sizing: {
              position: { ...fixedSizing.region, x: 0, y: 0 },
              padding: newPadding(0),
            },
            propsFn: () => ({
              dataRange: { x: { start: 0, end: 1 }, y: { start: 0, end: 1 } },
              includeWidthOfLastX: false,
            }),
            debug: false,
          }}
          components={components}
        />
      </div>
    )
  },
)
