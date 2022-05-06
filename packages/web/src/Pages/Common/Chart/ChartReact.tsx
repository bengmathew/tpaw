import React, {
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyRectSizingToHTMLElement,
  Padding,
  RectExt,
} from '../../../Utils/Geometry'
import {fGet} from '../../../Utils/Utils'
import {Chart, ChartAnimation, ChartXYRange} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'

export type ChartReactSizing = {position: RectExt; padding: Padding}
export type ChartReactState<Data> = {
  data: Data
  xyRange: ChartXYRange
  animation: ChartAnimation | null
}

export type ChartReactStatefull = {
  setSizing: (sizing: ChartReactSizing) => void
}
export const ChartReact = React.forwardRef(
  <Data,>(
    {
      state,
      starting,
      components,
    }: {
      state: ChartReactState<Data>
      starting: {
        sizing: ChartReactSizing
      }
      components: readonly ChartComponent<Data>[]
    },
    forwardedRef: React.ForwardedRef<ChartReactStatefull>
  ) => {
    const divRef = useRef<HTMLDivElement | null>(null)
    const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)

    const localChartRef = useRef<ChartReactStatefull | null>(null)

    useLayoutEffect(() => {
      if (!canvas) return
      applySizingToHTML(starting.sizing.position, fGet(divRef.current), canvas)
      // ignore startingSizing.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [canvas])

    useImperativeHandle(
      forwardedRef,
      () => ({
        setSizing: sizing => {
          if (!canvas) return
          applySizingToHTML(sizing.position, fGet(divRef.current), canvas)
          localChartRef.current?.setSizing(sizing)
        },
      }),
      [canvas]
    )

    return (
      <div className="absolute overflow-hidden" ref={divRef}>
        <canvas className="" style={{touchAction: 'none'}} ref={setCanvas} />
        {canvas && (
          <_AfterCanvas
            state={state}
            starting={starting}
            components={components}
            canvas={canvas}
            ref={localChartRef}
          />
        )}
      </div>
    )
  }
)

const applySizingToHTML = (
  position: RectExt,
  div: HTMLDivElement,
  canvas: HTMLCanvasElement
) => {
  const size = {width: position.width, height: position.height}
  applyRectSizingToHTMLElement(position, div)
  const dpr = window.devicePixelRatio || 1

  const pixelSize = (x: number) => Math.round(x * dpr)
  if (
    pixelSize(size.width) > canvas.width * 0.95 ||
    pixelSize(size.height) > canvas.height * 0.95
  ) {
    const enlarged = {
      width: Math.round(size.width * 1.25),
      height: Math.round(size.height * 1.25),
    }
    canvas.width = pixelSize(enlarged.width)
    canvas.height = pixelSize(enlarged.height)
    canvas.style.width = `${enlarged.width}px`
    canvas.style.height = `${enlarged.height}px`

    const ctx = fGet(canvas.getContext('2d'))
    ctx.scale(dpr, dpr)
  }
}

const _AfterCanvas = React.forwardRef(
  <Data,>(
    {
      state,
      starting,
      components,
      canvas,
    }: {
      state: ChartReactState<Data>
      starting: {
        sizing: ChartReactSizing
      }
      components: readonly ChartComponent<Data>[]
      canvas: HTMLCanvasElement
    },
    ref: React.ForwardedRef<ChartReactStatefull>
  ) => {
    const [chart, setChart] = useState<Chart<Data> | null>(null)
    useLayoutEffect(() => {
      const chart = new Chart(
        canvas,
        state.data,
        state.xyRange,
        [], // Will be set in useEffect
        _sizingTransform(starting.sizing)
      )
      setChart(chart)
      return () => chart.destroy()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useImperativeHandle(
      ref,
      () => ({
        setSizing: sizing => fGet(chart).setSizing(_sizingTransform(sizing)),
      }),
      [chart]
    )

    useEffect(() => {
      if (!chart) return
      chart.setState(state.data, state.xyRange, state.animation)
    }, [state, chart])

    useEffect(() => {
      if (!chart) return
      chart.setComponents(components)
    }, [components, chart])

    return <></>
  }
)

const _sizingTransform = ({
  position,
  padding,
}: {
  position: RectExt
  padding: Padding
}) => {
  const size = {width: position.width, height: position.height}
  return {size, padding}
}
