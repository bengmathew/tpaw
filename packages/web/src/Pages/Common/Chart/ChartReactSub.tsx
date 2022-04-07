import React, {useEffect, useLayoutEffect, useState} from 'react'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet} from '../../../Utils/Utils'
import {
  Chart,
  ChartAnimation,
  ChartPadding,
  ChartState,
  ChartXYRange,
} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {RectExt} from './ChartUtils/ChartUtils'

export type ChartReactState<Data> = {
  area: (bounds: {width: number; height: number}) => {
    viewport: RectExt
    padding: ChartPadding
  }
  data: Data
  xyRange: ChartXYRange
  alpha: number
  animation: ChartAnimation | null
}

export function ChartReactSub<Data>({
  state,
  components,
  animationForBoundsChange,
  canvas,
  bounds,
  debugName,
  registerDrawTarget,
  onDraw,
  registerAnimation,
}: {
  state: ChartReactState<Data>
  debugName: string
  components: readonly ChartComponent<Data>[]
  animationForBoundsChange: ChartAnimation | null
  canvas: HTMLCanvasElement
  bounds: {width: number; height: number}
  registerDrawTarget: (x: {
    draw: () => void
    pointerMoved: (position: {x: number; y: number} | null) => void
  }) => () => void
  onDraw: () => void
  registerAnimation: <T extends gsap.core.Tween | gsap.core.Timeline>(
    tween: T
  ) => T
}) {
  useAssertConst([onDraw])

  const [chart, setChart] = useState<Chart<Data> | null>(null)

  useLayoutEffect(() => {
    const chart = new Chart(
      canvas,
      _processState(state, bounds),
      components,
      debugName,
      onDraw,
      registerAnimation
    )
    const unregister = registerDrawTarget({
      draw: () => chart.draw(),
      pointerMoved: x => chart.pointerMoved(x),
    })
    setChart(chart)
    return () => unregister()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!chart) return
    chart.setState(_processState(state, bounds), state.animation)
    // Separate response for state and bounds.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state])

  useEffect(() => {
    if (!chart) return
    chart.setState(_processState(state, bounds), animationForBoundsChange)
    // Separate response for state and bounds.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds])

  useEffect(() => {
    if (!chart) return
    chart.setComponents(components)
  }, [components, chart])

  useEffect(() => {
    if (!chart) return
    return () => {
      fGet(chart).destroy()
    }
  }, [chart, debugName])

  return <></>
}

function _processState<Data>(
  {area, data, alpha, xyRange}: ChartReactState<Data>,
  {width, height}: {width: number; height: number}
): ChartState<Data> {
  const {padding, viewport} = area({width, height})
  return {
    viewport,
    padding,
    data,
    xyRange,
    alpha,
  }
}
