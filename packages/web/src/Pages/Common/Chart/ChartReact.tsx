import {localPoint} from '@visx/event'
import _ from 'lodash'
import React, {CSSProperties, useCallback, useEffect, useState} from 'react'
import Measure from 'react-measure'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {assert, fGet} from '../../../Utils/Utils'
import {ChartAnimation, ChartPadding, ChartXYRange} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {ChartReactSub} from './ChartReactSub'
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

type Info<Data extends any[]> = {
  [K in keyof Data]: {
    state: ChartReactState<Data[K]>
    components: readonly ChartComponent<Data[K]>[]
    key: string
    order: number
  }
}

export function ChartReact<Data extends any[]>({
  charts,
  animationForBoundsChange,
  className = '',
  style,
}: {
  charts: Info<Data>
  animationForBoundsChange: ChartAnimation | null
  className?: string
  style?: CSSProperties
}) {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)
  const [bounds, setBounds] = useState<{width: number; height: number} | null>(
    null
  )

  useEffect(() => {
    if (!bounds || !canvas) return
    canvas.style.width = `${bounds.width}px`
    canvas.style.height = `${bounds.height}px`
    const dpr = window.devicePixelRatio || 1
    canvas.width = bounds.width * dpr
    canvas.height = bounds.height * dpr
    const ctx = fGet(canvas.getContext('2d'))
    ctx.scale(dpr, dpr)
  }, [bounds, canvas])

  return (
    <div className={`${className}`} style={style}>
      <Measure
        bounds
        onResize={({bounds}) => {
          assert(bounds)
          setBounds({width: bounds.width, height: bounds.height})
        }}
      >
        {({measureRef}) => (
          <div className="w-full h-full" ref={measureRef}>
            {/* select-none because otherwise moving the chart pointer was
          selecting text  on mobile. Seems to work only on Android though. */}
            <div className=" select-none  absolute">
              <canvas
                style={{touchAction: 'none'}}
                ref={x => {
                  if (x && !canvas) setCanvas(x)
                }}
              />
              {bounds && canvas && (
                <_After
                  charts={charts}
                  animationForBoundsChange={animationForBoundsChange}
                  bounds={bounds}
                  canvas={canvas}
                />
              )}
            </div>
          </div>
        )}
      </Measure>
    </div>
  )
}

type _DrawTarget = {
  draw: () => void
  pointerMoved: (position: {x: number; y: number} | null) => void
  order: number
}
export function _After<Data extends any[]>({
  charts,
  animationForBoundsChange,
  bounds,
  canvas,
}: {
  charts: Info<Data>
  animationForBoundsChange: ChartAnimation | null
  bounds: {width: number; height: number}
  canvas: HTMLCanvasElement
}) {
  const [drawTargets] = useState(new Map<string, _DrawTarget>())

  const registerDrawTarget = useCallback(
    (key: string, value: _DrawTarget) => {
      drawTargets.set(key, value)
      return () => drawTargets.delete(key)
    },
    [drawTargets]
  )
  useAssertConst([drawTargets])

  useEffect(() => {
    const inHandler = (e: MouseEvent) =>
      drawTargets.forEach(x => x.pointerMoved(fGet(localPoint(canvas, e))))
    const outHandler = (e: MouseEvent) =>
      drawTargets.forEach(x => x.pointerMoved(fGet(localPoint(canvas, e))))
    canvas.addEventListener('pointermove', inHandler)
    canvas.addEventListener('pointerenter', inHandler)
    canvas.addEventListener('pointerleave', outHandler)
    return () => {
      canvas.removeEventListener('pointermove', inHandler)
      canvas.removeEventListener('pointerenter', inHandler)
      canvas.removeEventListener('pointerleave', outHandler)
    }
  }, [canvas, drawTargets])

  const onDraw = useCallback(() => {
    const ctx = fGet(canvas.getContext('2d'))
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    _.sortBy(Array.from(drawTargets.values()), x => x.order).forEach(x => {
      ctx.save()
      x.draw()
      ctx.restore()
    })
  }, [canvas, drawTargets])
  useAssertConst([canvas, drawTargets])

  return (
    <>
      {charts.map(({state, components, key, order}) => (
        <ChartReactSub
          key={key}
          debugName={key}
          state={state}
          components={components}
          animationForBoundsChange={animationForBoundsChange}
          bounds={bounds}
          canvas={canvas}
          registerDrawTarget={({draw, pointerMoved}) =>
            registerDrawTarget(key, {draw, pointerMoved, order})
          }
          onDraw={onDraw}
        />
      ))}{' '}
    </>
  )
}
