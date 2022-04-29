import React, {useEffect, useLayoutEffect, useState} from 'react'
import { Size, RectExt, Padding } from '../../../Utils/Geometry'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet} from '../../../Utils/Utils'
import {Chart, ChartAnimation, ChartXYRange} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'

export type ChartReactState<Data> = {
  area: (size: Size) => {viewport: RectExt; padding: Padding}
  data: Data
  xyRange: ChartXYRange
  alpha: number
  animation: ChartAnimation | null
}

export function ChartReactSub<Data>({
  state,
  components,
  canvas,
  startingSize,
  debugName,
  registerDrawTarget,
  onDraw,
  registerAnimation,
}: {
  state: ChartReactState<Data>
  debugName: string
  components: readonly ChartComponent<Data>[]
  canvas: HTMLCanvasElement
  startingSize: Size
  registerDrawTarget: (x: {
    draw: () => void
    pointerMoved: (position: {x: number; y: number}) => void
    setSize: (size: Size) => void
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
      state.data,
      {area: state.area, xyRange: state.xyRange, alpha: state.alpha},
      components,
      debugName,
      startingSize,
      registerAnimation
    )
    const unregister = registerDrawTarget({
      draw: () => chart.draw(),
      pointerMoved: x => chart.updatePointer(x),
      setSize: size => chart.setSize(size),
    })
    setChart(chart)
    return () => unregister()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!chart) return
    chart.updateState(
      state.data,
      {area: state.area, xyRange: state.xyRange, alpha: state.alpha},
      state.animation
    )
    onDraw()
    // Separate response for state and size.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state])

  useEffect(() => {
    if (!chart) return
    chart.setComponents(components)
    onDraw()
  }, [components, chart, onDraw])
  useAssertConst([onDraw])

  useEffect(() => {
    if (!chart) return
    return () => {
      fGet(chart).destroy()
    }
  }, [chart, debugName])

  return <></>
}
