import gsap from 'gsap'
import React, {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  Padding,
  RectExt,
  applySizeToHTMLElement,
  originCSSStyle,
} from '../../../Utils/Geometry'
import { fGet } from '../../../Utils/Utils'
import { Chart, ChartAnimation } from './Chart'
import { ChartComponent } from './ChartComponent/ChartComponent'
import { interpolate } from '../../../Utils/Interpolate'

export type ChartReactSizing = { position: RectExt; padding: Padding }
export type ChartReactState<Data> = {
  data: Data
  animation: ChartAnimation | null
}

export type ChartReactStatefull<Data> = {
  setSizing: (
    sizing: ChartReactSizing,
    animation: ChartAnimation | null,
  ) => void
  setData: (data: Data, animation: ChartAnimation | null) => void
  getData: () => Data
  onPointerMove: (e: React.PointerEvent | React.TouchEvent) => void
  onPointerEnter: (e: React.PointerEvent | React.TouchEvent) => void
  onPointerLeave: (e: React.PointerEvent | React.TouchEvent) => void
}
export const ChartReact = React.forwardRef(
  <Data,>(
    {
      starting,
      components,
      onHover: onHoverIn,
      captureTouchEvents = false,
    }: {
      starting: {
        data: Data
        sizing: ChartReactSizing
        propsFn: Chart<Data>['propsFn']
        debug?: boolean
      }
      onHover?: (hover: boolean) => void
      components: () => readonly ChartComponent<Data>[]
      captureTouchEvents?: boolean
    },
    forwardedRef: React.ForwardedRef<ChartReactStatefull<Data>>,
  ) => {
    const divRef = useRef<HTMLDivElement>(null)
    const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)
    const [chart, setChart] = useState<_AfterCanvasStateful<Data> | null>(null)

    const onHoverRef = useRef(onHoverIn)
    onHoverRef.current = onHoverIn
    const onHover = useCallback(
      (hover: boolean) => onHoverRef.current?.(hover),
      [],
    )

    const [sizingTransition, setSizingTransition] = useState({
      from: starting.sizing,
      target: starting.sizing,
      animation: null as ChartAnimation | null,
    })

    useEffect(() => {
      if (canvas) {
        applyTargetSizingToHTML(
          sizingTransition.target.position,
          sizingTransition.animation?.duration ?? 0,
          fGet(divRef.current),
          canvas,
          'update',
        )
      }

      const obj = { progress: 0 }
      const tween = gsap.to(obj, {
        duration: (sizingTransition.animation?.duration ?? 0) / 1000,
        progress: 1,
        ease: sizingTransition.animation?.ease,
        onUpdate: function () {
          if (!canvas || !chart) return
          const obj = this.targets()[0] as { progress: number }
          chart.setSizing(
            interpolate({
              from: sizingTransition.from,
              target: sizingTransition.target,
              progress: obj.progress,
            }),
          )
        },
      })
      return () => {
        tween.kill()
      }
    }, [canvas, chart, sizingTransition])

    useLayoutEffect(() => {
      if (!canvas) return
      applyTargetSizingToHTML(
        starting.sizing.position,
        10,
        fGet(divRef.current),
        canvas,
        'init',
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
          setSizing: (sizing, animation) => {
            setSizingTransition((old) => ({
              from: old.target,
              target: sizing,
              animation,
            }))
          },
          setData: (data, animation) => chart.setData(data, animation),
          getData: () => chart.getData(),
          onPointerMove: (e) => chart.onPointerMove(e),
          onPointerEnter: (e) => chart.onPointerEnter(e),
          onPointerLeave: (e) => chart.onPointerLeave(e),
        }
      },
      [canvas, chart],
    )

    return (
      <div
        // select-none prevents long press on mobile.
        className="absolute overflow-hidden select-none"
        ref={divRef}
        style={{ ...originCSSStyle({ x: 0, y: 0 }) }}
        onTouchStart={
          captureTouchEvents ? (e) => chart?.onPointerEnter(e) : undefined
        }
        onTouchMove={
          captureTouchEvents ? (e) => chart?.onPointerMove(e) : undefined
        }
        onTouchEnd={
          captureTouchEvents ? (e) => chart?.onPointerLeave(e) : undefined
        }
      >
        <canvas className="" style={{ touchAction: 'none' }} ref={setCanvas} />
        {canvas && (
          <_AfterCanvas
            starting={{
              ...starting,
              onHover,
              canvas,
            }}
            components={components}
            ref={setChart}
          />
        )}
      </div>
    )
  },
)

const applyTargetSizingToHTML = (
  position: RectExt,
  duration: number,
  div: HTMLDivElement,
  canvas: HTMLCanvasElement,
  type: 'init' | 'update',
) => {
  const size = { width: position.width, height: position.height }
  div.style.width = `${position.width}px`
  div.style.height = `${position.height}px`
  div.style.transform = `translate(${position.x}px, ${position.y}px)`
  div.style.transitionProperty = 'width, height, transform'
  div.style.transitionDuration = `${duration}ms`
  applySizeToHTMLElement(position, div)
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

type _AfterCanvasStateful<Data> = {
  setSizing: (sizing: ChartReactSizing) => void
  setData: (data: Data, animation: ChartAnimation | null) => void
  getData: () => Data
  onPointerMove: (e: React.PointerEvent | React.TouchEvent) => void
  onPointerEnter: (e: React.PointerEvent | React.TouchEvent) => void
  onPointerLeave: (e: React.PointerEvent | React.TouchEvent) => void
}

const _AfterCanvas = React.forwardRef(
  <Data,>(
    {
      starting,
      components,
    }: {
      starting: {
        canvas: HTMLCanvasElement
        data: Data
        sizing: ChartReactSizing
        propsFn: Chart<Data>['propsFn']
        debug?: boolean
        onHover: (hover: boolean) => void
      }
      components: () => readonly ChartComponent<Data>[]
    },
    ref: React.ForwardedRef<_AfterCanvasStateful<Data>>,
  ) => {
    const [chart, setChart] = useState<Chart<Data> | null>(null)
    useLayoutEffect(() => {
      const chart = new Chart(
        starting.canvas,
        starting.onHover,
        starting.data,
        [], // Will be set in useEffect
        _sizingTransform(starting.sizing),
        starting.propsFn,
        starting.debug ?? false,
      )
      setChart(chart)
      return () => chart.destroy()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useImperativeHandle(
      ref,
      () => {
        if (!chart) return null as unknown as _AfterCanvasStateful<Data>
        // Need to check isDestroyed because in dev during double initialization
        // chart was destroyed, but ref was still called which led to problems.
        return {
          setSizing: (sizing) =>
            !chart.isDestroyed && chart.setSizing(_sizingTransform(sizing)),
          setData: (data, animation) =>
            !chart.isDestroyed && chart.setParams(data, animation),
          getData: () => chart.getParams(),
          onPointerMove: (e) => !chart.isDestroyed && chart.onPointerMove(e),
          onPointerEnter: (e) => !chart.isDestroyed && chart.onPointerEnter(e),
          onPointerLeave: (e) => !chart.isDestroyed && chart.onPointerLeave(e),
        }
      },
      [chart],
    )

    useEffect(() => {
      // Need to check isDestroyed because in dev during double initialization
      // chart was destroyed, but ref was still called which led to problems.
      if (!chart || chart.isDestroyed) return
      chart.setComponents(components())
    }, [components, chart])

    return <></>
  },
)

const _sizingTransform = ({
  position,
  padding,
}: {
  position: RectExt
  padding: Padding
}) => {
  const size = { width: position.width, height: position.height }
  return { size, padding }
}
