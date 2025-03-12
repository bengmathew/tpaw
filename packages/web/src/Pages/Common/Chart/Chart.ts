import { block, linearFnFomPoints } from '@tpaw/common'
import { localPoint } from '@visx/event'
import { gsap } from 'gsap'
import _ from 'lodash'
import { Padding, Rect, Size, XY, rectExt } from '../../../Utils/Geometry'
import { interpolate } from '../../../Utils/Interpolate'
import { SimpleRange } from'@tpaw/common'
import { Transition, transitionTransform } from '../../../Utils/Transition'
import { assert, fGet } from '../../../Utils/Utils'
import {
  ChartComponent,
  ChartComponentPointerTargetY,
} from './ChartComponent/ChartComponent'
import { ChartContext } from './ChartContext'

export type ChartDataRange = { x: SimpleRange; y: SimpleRange }
export type ChartAnimation = { ease: gsap.EaseFunction; duration: number }
export type ChartSizing = { size: Size; padding: Padding }
export type ChartAnimatedProperty<T> = {
  transition: Transition<T>
  animation: gsap.core.Animation | null
}

const pointerAnimationDuration = 1

type _Props = {
  dataRange: ChartDataRange
  includeWidthOfLastX: boolean
}

export class Chart<Params> {
  private _destroyed = false

  private _params: ChartAnimatedProperty<Params>
  private _pointer: {
    // Visual position will not always line up to data grid, or be within data
    // bounds because of scale is invertible only modulo floating point precision.
    // So the guarantee of visual position is that the target will be "close" to
    // the data grid and within the visual plot area bounds.
    visualPosition: ChartAnimatedProperty<XY>
    hover: ChartAnimatedProperty<0 | 1>
    // press: ChartAnimatedProperty<0 | 1>
  }

  private _callbacks = {
    draw: () => this.draw('draw'),
    registerAnimation: this._registerAnimation.bind(this),
  }
  private _activeAnimations: gsap.core.Animation[] = []

  constructor(
    private _canvas: HTMLCanvasElement,
    private _onHover: (hover: boolean) => void,
    params: Params,
    private _components: readonly ChartComponent<Params>[],
    private _sizing: ChartSizing,
    private propsFn: (params: Params) => _Props,
    private _debug: boolean,
  ) {
    this._params = {
      transition: { target: params, from: params, progress: 1 },
      animation: null,
    }
    const props = this.propsFn(params)

    this._pointer = {
      visualPosition: block(() => {
        const target = _snapPointerToGridAndPlotArea(
          { x: this._sizing.size.width * 0.25, y: 0 },
          _chartDerivedState(this._sizing, props),
        )
        return {
          transition: { from: target, target, progress: 1 },
          animation: null,
        }
      }),
      hover: {
        transition: { from: 0, target: 0, progress: 1 },
        animation: null,
      },
    }

    this.draw('init')
  }
  onPointerMove(e: React.PointerEvent | React.TouchEvent) {
    this._handlePointer(e, true)
  }
  onPointerEnter(e: React.PointerEvent | React.TouchEvent) {
    this._handlePointer(e, true)
  }
  onPointerLeave(e: React.PointerEvent | React.TouchEvent) {
    this._handlePointer(e, false)
  }

  get isDestroyed() {
    return this._destroyed
  }
  destroy() {
    this._destroyed = true
    this._components.forEach((x) => x.destroy?.())
    this._params.animation?.kill()
    this._pointer.visualPosition.animation?.kill()
    this._pointer.hover.animation?.kill()
    assert(this._activeAnimations.length === 0)
  }

  setComponents(components: readonly ChartComponent<Params>[]) {
    this._components.forEach((x) => x.destroy?.())
    this._components = components
    this.draw('init')
  }

  getParams() {
    return this._params.transition.target
  }

  setParams(params: Params, animation: ChartAnimation | null) {
    if (this._params.transition.target === params) return

    this._params.transition.from = this._params.transition.target
    this._params.transition.target = params
    this._params.animation?.kill()
    if (animation) {
      this._params.animation = this._registerAnimation(
        gsap.fromTo(
          this._params.transition,
          { progress: 0 },
          {
            progress: 1,
            ease: animation.ease,
            duration: animation.duration,
          },
        ),
      )
    } else {
      this._params.animation = null
      this._params.transition.progress = 1
    }

    const { stateTransition } = this._getContext()

    this._pointer.visualPosition.animation?.kill()
    this._pointer.visualPosition.transition = {
      from: interpolate(this._pointer.visualPosition.transition),
      target: _snapPointerToGridAndPlotArea(
        this._pointer.visualPosition.transition.target,
        stateTransition.target.derivedState,
      ),
      progress: 0,
    }
    this._pointer.visualPosition.animation = this._registerAnimation(
      gsap.to(this._pointer.visualPosition.transition, {
        progress: 1,
        duration: pointerAnimationDuration,
        ease: 'power4',
      }),
    )
    this.draw('stateAndPointer')
  }

  setSizing(sizing: ChartSizing) {
    if (_.isEqual(sizing, this._sizing)) return
    this._sizing = sizing

    const { stateTransition } = this._getContext()
    this._pointer.visualPosition.animation?.kill()
    const target = _snapPointerToGridAndPlotArea(
      this._pointer.visualPosition.transition.target,
      stateTransition.target.derivedState,
    )
    this._pointer.visualPosition.transition = {
      from: target,
      target,
      progress: 1,
    }

    this.draw('sizingAndPointer')
  }

  draw(
    reason:
      | 'draw'
      | 'init'
      | 'pointer'
      | 'stateAndPointer'
      | 'sizingAndPointer' = 'draw',
  ) {
    assert(!this._destroyed)
    const context = this._getContext()

    const { canvasContext } = context
    canvasContext.clearRect(0, 0, this._canvas.width, this._canvas.height)
    const pointerTargetYs: (ChartComponentPointerTargetY | null)[] = []
    this._components.forEach((component) => {
      canvasContext.save()
      pointerTargetYs.push(
        component.draw(
          context,
          reason,
          this._callbacks.registerAnimation,
          (dataX: number) =>
            _.flatten(_.compact(pointerTargetYs).map((x) => x(dataX))),
        ),
      )
      canvasContext.restore()
    })
  }

  private _getContext(): ChartContext<Params> {
    // Without this we will have to explicity ignore the from state when
    // progress is 1, for example calculating the xRange.
    if (this._params.transition.progress === 1) {
      this._params.transition.from = this._params.transition.target
    }
    const stateTransition = transitionTransform(
      this._params.transition,
      (params) => {
        const props = this.propsFn(params)
        return {
          params,
          dataRange: props.dataRange,
          includeWidthOfLastX: props.includeWidthOfLastX,
          derivedState: _chartDerivedState(this._sizing, props),
        }
      },
    )

    const currState = block((): ChartContext<Params>['currState'] => {
      const dataRange = interpolate(
        transitionTransform(stateTransition, (x) => x.dataRange),
      )
      const dataRangeUnion = {
        x: SimpleRange.union(
          stateTransition.from.dataRange.x,
          stateTransition.target.dataRange.x,
        ),
        y: SimpleRange.union(
          stateTransition.from.dataRange.y,
          stateTransition.target.dataRange.y,
        ),
      }
      const includeWidthOfLastX = stateTransition.target.includeWidthOfLastX
      const derivedState = _chartDerivedState(this._sizing, {
        dataRange,
        includeWidthOfLastX,
      })
      const dataXRangeCurrentlyVisible = {
        start: Math.ceil(
          derivedState.scale.x.inverse.notRounded(derivedState.plotArea.x),
        ),
        end: Math.floor(
          derivedState.scale.x.inverse.notRounded(derivedState.plotArea.right),
        ),
      }
      const pointer = {
        position: block(() => {
          const visual = _clampToPlotArea(
            interpolate(this._pointer.visualPosition.transition),
            derivedState.plotArea,
          )
          return {
            visual,
            dataNotRounded: {
              x: SimpleRange.Closed.clamp(
                derivedState.scale.x.inverse.notRounded(visual.x),
                dataRangeUnion.x,
              ),
            },
          }
        }),
        hover: interpolate(this._pointer.hover.transition),
      }
      return {
        dataRangeUnion,
        dataXRangeCurrentlyVisible,
        pointer,
        derivedState,
      }
    })
    return {
      canvasContext: fGet(this._canvas.getContext('2d')),
      stateTransition,
      pointerTransition: {
        visualPosition: this._pointer.visualPosition.transition,
        hover: this._pointer.hover.transition,
      },
      sizing: this._sizing,
      currState,
    }
  }

  private _handlePointer(
    e: React.PointerEvent | React.TouchEvent,
    isInside: boolean,
  ) {
    const { stateTransition } = this._getContext()

    const positionFromEvent = fGet(localPoint(this._canvas, e))
    // Hover
    const resetVisualPosition = block(() => {
      const target = isInside ? 1 : 0
      if (target !== this._pointer.hover.transition.target) {
        this._onHover(target === 1)
        this._pointer.hover.animation?.kill()
        this._pointer.hover.transition = {
          from: interpolate(this._pointer.hover.transition),
          target,
          progress: 0,
        }
        this._pointer.hover.animation = this._registerAnimation(
          gsap.to(this._pointer.hover.transition, {
            progress: 1,
            duration: pointerAnimationDuration,
            ease: 'power4',
          }),
        )
        return target === 1
      } else {
        return false
      }
    })
    // Visual Position
    {
      this._pointer.visualPosition.animation?.kill()
      const target = _snapPointerToGridAndPlotArea(
        positionFromEvent,
        stateTransition.target.derivedState,
      )
      this._pointer.visualPosition.transition = {
        from: resetVisualPosition
          ? target
          : interpolate(this._pointer.visualPosition.transition),
        target: target,
        progress: 0,
      }
      this._pointer.visualPosition.animation = this._registerAnimation(
        gsap.to(this._pointer.visualPosition.transition, {
          progress: 1,
          duration: pointerAnimationDuration,
          ease: 'power4',
        }),
      )
    }

    this.draw('pointer')
  }

  private _registerAnimation<T extends gsap.core.Tween | gsap.core.Timeline>(
    tween: T,
  ): T {
    tween.eventCallback('onStart', () => {
      this._activeAnimations.push(tween)
      if (this._activeAnimations.length === 1) {
        gsap.ticker.add(this._callbacks.draw)
      }
    })
    const handleDone = () => {
      _.remove(this._activeAnimations, (x) => x === tween)
      if (this._activeAnimations.length === 0) {
        gsap.ticker.remove(this._callbacks.draw)
      }
    }
    tween.eventCallback('onComplete', handleDone)
    // As per https://greensock.com/forums/topic/22563-working-example-of-oninterrupt-callback/
    // this obviates the need to handle "kill"
    tween.eventCallback('onInterrupt', handleDone)
    return tween
  }
}

export type ChartStateDerived = ReturnType<typeof _chartDerivedState>
const _chartDerivedState = (
  sizing: ChartSizing,
  { dataRange, includeWidthOfLastX }: _Props,
) => {
  const { size, padding } = sizing
  const viewport = rectExt({ x: 0, y: 0, ...size })

  const plotArea = rectExt({
    x: viewport.x + padding.left,
    y: viewport.y + padding.top,
    width: viewport.width - padding.left - padding.right,
    height: viewport.height - padding.top - padding.bottom,
  })

  const basicScaleX = linearFnFomPoints(
    dataRange.x.start,
    plotArea.x,
    dataRange.x.end + (includeWidthOfLastX ? 1 : 0),
    plotArea.width + plotArea.x,
  )
  const scaleX = block(() => {
    const start = (dataX: number) => basicScaleX(dataX)
    const end = (dataX: number) => basicScaleX(dataX + 1)
    const width = start(1) - start(0)
    const center = (dataX: number) => start(dataX) + width / 2
    const inverse = {
      rounded: (pixelX: number) => Math.round(basicScaleX.inverse(pixelX)),
      notRounded: (pixelX: number) => basicScaleX.inverse(pixelX),
    }
    return { start, end, center, inverse }
  })

  // Data to pixel coordinates.
  const scale = {
    x: scaleX,
    y: linearFnFomPoints(
      dataRange.y.start,
      plotArea.y + plotArea.height,
      // Handle case where dataRange.y.end === dataRange.y.start.
      Math.max(dataRange.y.start + 0.0000001, dataRange.y.end),
      plotArea.y,
    ),
  }
  return { plotArea, scale, viewport, padding }
}

const _snapPointerToGridAndPlotArea = (
  pointerInPixelsCoord: XY,
  { scale, plotArea }: ChartStateDerived,
) =>
  _clampToPlotArea(
    {
      x: scale.x.start(scale.x.inverse.rounded(pointerInPixelsCoord.x)),
      y: pointerInPixelsCoord.y,
    },
    plotArea,
  )

const _clampToPlotArea = ({ x, y }: XY, plotArea: Rect) => ({
  x: _.clamp(x, plotArea.x, plotArea.x + plotArea.width),
  y: _.clamp(y, plotArea.y, plotArea.y + plotArea.height),
})
