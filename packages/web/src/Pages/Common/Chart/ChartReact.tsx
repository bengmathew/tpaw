import {localPoint} from '@visx/event'
import {gsap} from 'gsap'
import _ from 'lodash'
import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyRectSizingToHTMLElement,
  RectExt,
  Size,
} from '../../../Utils/Geometry'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet} from '../../../Utils/Utils'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {ChartReactState, ChartReactSub} from './ChartReactSub'

type Info<Data extends any[]> = {
  [K in keyof Data]: {
    state: ChartReactState<Data[K]>
    components: readonly ChartComponent<Data[K]>[]
    key: string
    order: number
  }
}

export type ChartRef = {setPosition: (position: RectExt) => void}
export function ChartReact<Data extends any[]>({
  charts,
  startingPosition,
  chartRef,
}: {
  charts: Info<Data>
  startingPosition: RectExt
  chartRef: (chart: ChartRef) => void
}) {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null)

  const divRef = useRef<HTMLDivElement | null>(null)
  const localChartRef = useRef<{setSize: (size: Size) => void} | null>(null)
  useLayoutEffect(
    () => {
      const div = fGet(divRef.current)
      applyRectSizingToHTMLElement(startingPosition, div)
      chartRef({
        setPosition: position => {
          applyRectSizingToHTMLElement(position, div)
          localChartRef.current?.setSize(position)
        },
      })
    },
    // exclude: chartRef, canvas, startingSize
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  )
  return (
    <div className="absolute overflow-hidde" ref={divRef}>
      <canvas
        className=""
        style={{touchAction: 'none'}}
        ref={x => {
          if (!x || x === canvas) return
          const dpr = window.devicePixelRatio || 1
          const canvasSize = {
            width: startingPosition.width * 1.25,
            height: startingPosition.height * 1.25,
          }
          x.width = canvasSize.width * dpr
          x.height = canvasSize.height * dpr
          x.style.width = `${canvasSize.width}px`
          x.style.height = `${canvasSize.height}px`
          fGet(x.getContext('2d')).scale(dpr, dpr)
          setCanvas(x)
        }}
      />
      {canvas && (
        <_After
          charts={charts}
          startingSize={{
            width: startingPosition.width,
            height: startingPosition.height,
          }}
          canvas={canvas}
          chartRef={x => (localChartRef.current = x)}
        />
      )}
    </div>
  )
}

type _DrawTarget = {
  draw: () => void
  pointerMoved: (position: {x: number; y: number}) => void
  setSize: (size: Size) => void
  order: number
}

export function _After<Data extends any[]>({
  charts,
  startingSize,
  canvas,
  chartRef,
}: {
  charts: Info<Data>
  startingSize: Size
  canvas: HTMLCanvasElement
  chartRef: (chart: {setSize: (size: Size) => void}) => void
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

  useEffect(() => {
    const handler = (e: MouseEvent) =>
      drawTargets.forEach(x => {
        x.pointerMoved(fGet(localPoint(canvas, e)))
        onDraw()
      })

    canvas.addEventListener('pointermove', handler)
    canvas.addEventListener('pointerenter', handler)
    canvas.addEventListener('pointerleave', handler)
    return () => {
      canvas.removeEventListener('pointermove', handler)
      canvas.removeEventListener('pointerenter', handler)
      canvas.removeEventListener('pointerleave', handler)
    }
  }, [canvas, drawTargets, onDraw])
  useAssertConst([canvas, drawTargets, onDraw])
  useAssertConst([onDraw, drawTargets, canvas])


  useLayoutEffect(() => {
    chartRef({
      setSize: size => {
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

        drawTargets.forEach(x => x.setSize(size))
        onDraw()
      },
    })
    // exclude: chartRef
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onDraw, drawTargets, canvas])
  useAssertConst([onDraw, drawTargets, canvas])

  const [activeAnimations] = useState<(gsap.core.Tween | gsap.core.Timeline)[]>(
    []
  )

  const registerAnimation = useCallback(
    <T extends gsap.core.Tween | gsap.core.Timeline>(tween: T): T => {
      tween.eventCallback('onStart', () => {
        activeAnimations.push(tween)
        if (activeAnimations.length === 1) {
          gsap.ticker.add(onDraw)
        }
      })
      const handleDone = () => {
        _.remove(activeAnimations, x => x === tween)
        if (activeAnimations.length === 0) {
          gsap.ticker.remove(onDraw)
        }
      }
      tween.eventCallback('onComplete', handleDone)
      // As per https://greensock.com/forums/topic/22563-working-example-of-oninterrupt-callback/
      // this obviates the need to handle "kill"
      tween.eventCallback('onInterrupt', handleDone)
      return tween
    },
    [activeAnimations, onDraw]
  )
  useAssertConst([activeAnimations, onDraw])

  return (
    <>
      {charts.map(({state, components, key, order}) => (
        <ChartReactSub
          key={key}
          debugName={key}
          state={state}
          components={components}
          startingSize={startingSize}
          canvas={canvas}
          registerDrawTarget={({draw, pointerMoved, setSize}) =>
            registerDrawTarget(key, {draw, pointerMoved, setSize, order})
          }
          onDraw={onDraw}
          registerAnimation={registerAnimation}
        />
      ))}{' '}
    </>
  )
}
