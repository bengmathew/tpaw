import React, {
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyRegionToHTMLElement,
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

export type ChartReactStatefull<Data> = {
  setSizing: (sizing: ChartReactSizing) => void
  setState: (
    data: Data,
    xyRange: ChartXYRange,
    animation: ChartAnimation | null
  ) => void
  getState: () => {data: Data; xyRange: ChartXYRange}
}
export const ChartReact = React.forwardRef(
  <Data,>(
    {
      starting,
      components,
    }: {
      starting: {
        data: Data
        xyRange: ChartXYRange
        sizing: ChartReactSizing
      }
      components: ()=> readonly ChartComponent<Data>[]
    },
    forwardedRef: React.ForwardedRef<ChartReactStatefull<Data>>
  ) => {
    const divRef = useRef<HTMLDivElement>(null)
    const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)
    const [chart, setChart] = useState<ChartReactStatefull<Data> | null>(null)

    useLayoutEffect(() => {
      if (!canvas) return
      applySizingToHTML(
        starting.sizing.position,
        fGet(divRef.current),
        canvas,
        'init'
      )
      // ignore startingSizing.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [canvas])

    useImperativeHandle(
      forwardedRef,
      () => {
        if (!canvas || !chart)
          return null as unknown as ChartReactStatefull<Data>
        return {
          setSizing: sizing => {
            if (!canvas) return
            applySizingToHTML(
              sizing.position,
              fGet(divRef.current),
              canvas,
              'update'
            )
            chart.setSizing(sizing)
          },
          setState: (data, xyRange, animation) => {
            chart.setState(data, xyRange, animation)
          },
          getState: () => chart.getState(),
        }
      },
      [canvas, chart]
    )

    return (
      <div className="absolute overflow-hidden" ref={divRef}>
        <canvas className="" style={{touchAction: 'none'}} ref={setCanvas} />
        {canvas && (
          <_AfterCanvas
            starting={starting}
            components={components}
            canvas={canvas}
            ref={setChart}
          />
        )}
      </div>
    )
  }
)

const applySizingToHTML = (
  position: RectExt,
  div: HTMLDivElement,
  canvas: HTMLCanvasElement,
  type: 'init' | 'update'
) => {
  const size = {width: position.width, height: position.height}
  applyRegionToHTMLElement(position, div)
  const dpr = window.devicePixelRatio || 1

  const pixelSize = (x: number) => Math.round(x * dpr)
  if (
    type === 'init' ||
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
      starting,
      components,
      canvas,
    }: {
      starting: {
        data: Data
        xyRange: ChartXYRange
        sizing: ChartReactSizing
      }
      components: ()=>readonly ChartComponent<Data>[]
      canvas: HTMLCanvasElement
    },
    ref: React.ForwardedRef<ChartReactStatefull<Data>>
  ) => {
    const [chart, setChart] = useState<Chart<Data> | null>(null)
    useLayoutEffect(() => {
      const chart = new Chart(
        canvas,
        starting.data,
        starting.xyRange,
        [], // Will be set in useEffect
        _sizingTransform(starting.sizing)
      )
      setChart(chart)
      return () => chart.destroy()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useImperativeHandle(
      ref,
      () => {
        if (!chart) return null as unknown as ChartReactStatefull<Data>
        return {
          setSizing: sizing => chart.setSizing(_sizingTransform(sizing)),
          setState: (data, xyRange, animation) =>
            chart.setState(data, xyRange, animation),
          getState: () => chart.getState(),
        }
      },
      [chart]
    )

    useEffect(() => {
      if (!chart) return
      chart.setComponents(components())
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
