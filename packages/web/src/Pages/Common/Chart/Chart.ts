import { linearFnFomPoints } from '@tpaw/common'
import { gsap } from 'gsap'
import _ from 'lodash'
import { Padding, rectExt, Size, XY } from '../../../Utils/Geometry'
import { interpolate } from '../../../Utils/Interpolate'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { assert, fGet } from '../../../Utils/Utils'
import { ChartComponent } from './ChartComponent/ChartComponent'
import { ChartContext } from './ChartContext'
import { Transition } from '../../../Utils/Transition'
import { localPoint } from '@visx/event'

export type ChartXYRange = { x: SimpleRange; y: SimpleRange }
export type ChartAnimation = { ease: gsap.EaseFunction; duration: number }
export type ChartState = { xyRange: ChartXYRange }
export type ChartSizing = { size: Size; padding: Padding }

const pointerAnimationDuration = 1

type _Animated<T> = {
  transition: Transition<T>
  animation: gsap.core.Animation | null
}
export class Chart<Data> {
  private _canvas: HTMLCanvasElement
  private _dataTransition: Transition<Data>
  private _dataAnimation: gsap.core.Animation | null = null
  private _stateTransition: Transition<ChartState>
  private _stateAnimation: gsap.core.Animation | null = null
  private _sizing: ChartSizing
  private _pointerInDataCoord: _Animated<XY>

  private _components: readonly ChartComponent<Data>[]

  private _callbacks = {
    draw: this.draw.bind(this),
    registerAnimation: this._registerAnimation.bind(this),
    onMouse: this._handleMouse.bind(this),
  }
  private _activeAnimations: gsap.core.Animation[] = []

  constructor(
    canvas: HTMLCanvasElement,
    data: Data,
    xyRange: ChartXYRange,
    components: readonly ChartComponent<Data>[],
    sizing: ChartSizing,
  ) {
    this._canvas = canvas
    this._components = components
    this._sizing = sizing

    this._dataTransition = { target: data, from: data, progress: 1 }
    this._stateTransition = {
      target: { xyRange },
      from: { xyRange },
      progress: 1,
    }

    this._pointerInDataCoord = (() => {
      const target = _getPointerInDataCoord(
        { x: this._sizing.size.width * 0.25, y: 0 },
        _chartDerivedState(this._sizing, { xyRange }),
      )
      return {
        transition: { from: target, target, progress: 1 },
        animation: null,
      }
    })()

    this._components.forEach(
      (x) =>
        x.update?.(
          'init',
          this._getContext(),
          this._callbacks.registerAnimation,
        ),
    )
    canvas.addEventListener('pointermove', this._callbacks.onMouse)
    canvas.addEventListener('pointerenter', this._callbacks.onMouse)
    canvas.addEventListener('pointerleave', this._callbacks.onMouse)
  }

  destroy() {
    this._components.forEach((x) => x.destroy?.())
    this._dataAnimation?.kill()
    this._stateAnimation?.kill()
    this._pointerInDataCoord.animation?.kill()
    assert(this._activeAnimations.length === 0)
    this._canvas.removeEventListener('pointermove', this._callbacks.onMouse)
    this._canvas.removeEventListener('pointerenter', this._callbacks.onMouse)
    this._canvas.removeEventListener('pointerleave', this._callbacks.onMouse)
  }

  setComponents(components: readonly ChartComponent<Data>[]) {
    this._components.forEach((x) => x.destroy?.())
    this._components = components
    const context = this._getContext()
    this._components.forEach(
      (x) => x.update?.('init', context, this._callbacks.registerAnimation),
    )
    this.draw()
  }

  getState() {
    return {
      data: this._dataTransition.target,
      xyRange: this._stateTransition.target.xyRange,
    }
  }

  setState(
    data: Data,
    xyRange: ChartXYRange,
    animation: ChartAnimation | null,
  ) {
    if (this._dataTransition.target !== data) {
      // Cannot interpolate data, so accept that data update mid transition will
      // be jerky.
      this._dataTransition.from = this._dataTransition.target
      this._dataTransition.target = data
      this._dataAnimation?.kill()
      this._dataAnimation = null
      if (animation) {
        this._dataAnimation = this._registerAnimation(
          gsap.fromTo(
            this._dataTransition,
            { progress: 0 },
            {
              progress: 1,
              ease: animation.ease,
              duration: animation.duration,
            },
          ),
        )
      } else {
        this._dataTransition.progress = 1
      }
    }

    this._stateTransition.from = interpolate(this._stateTransition)
    this._stateTransition.target = { xyRange }
    this._stateAnimation?.kill()
    if (animation) {
      this._stateAnimation = this._registerAnimation(
        gsap.fromTo(
          this._stateTransition,
          { progress: 0 },
          { progress: 1, ease: animation.ease, duration: animation.duration },
        ),
      )
    } else {
      this._stateTransition.progress = 1
    }

    this._pointerInDataCoord.animation?.kill()
    this._pointerInDataCoord.transition = {
      from: interpolate(this._pointerInDataCoord.transition),
      target: {
        x: _.clamp(
          this._pointerInDataCoord.transition.target.x,
          xyRange.x.start,
          xyRange.x.end,
        ),
        y: _.clamp(
          this._pointerInDataCoord.transition.target.y,
          xyRange.y.start,
          xyRange.y.end,
        ),
      },
      progress: 0,
    }
    this._pointerInDataCoord.animation = this._registerAnimation(
      gsap.to(this._pointerInDataCoord.transition, {
        progress: 1,
        duration: pointerAnimationDuration,
        ease: 'power4',
      }),
    )

    this._components.forEach(
      (x) =>
        x.update?.(
          'stateAndPointer',
          this._getContext(),
          this._callbacks.registerAnimation,
        ),
    )
    this.draw()
  }

  setSizing(sizing: ChartSizing) {
    if (_.isEqual(sizing, this._sizing)) return

    this._sizing = sizing
    const context = this._getContext()
    this._components.forEach(
      (x) => x.update?.('sizing', context, this._callbacks.registerAnimation),
    )
    this.draw()
  }

  draw() {
    const context = this._getContext()
    const { canvasContext } = context
    canvasContext.clearRect(0, 0, this._canvas.width, this._canvas.height)
    this._components.forEach((component) => {
      canvasContext.save()
      component.draw(context)
      canvasContext.restore()
    })
  }

  private _getContext(): ChartContext<Data> {
    const currState = interpolate(this._stateTransition)
    return {
      canvasContext: fGet(this._canvas.getContext('2d')),
      dataTransition: this._dataTransition,
      stateTransition: this._stateTransition,
      pointerInDataCoordTransition: this._pointerInDataCoord.transition,
      currState,
      currPointerInDataCoord: interpolate(this._pointerInDataCoord.transition),
      derivedState: {
        prev: _chartDerivedState(this._sizing, this._stateTransition.from),
        target: _chartDerivedState(this._sizing, this._stateTransition.target),
        curr: _chartDerivedState(this._sizing, currState),
      },
      sizing: this._sizing,
    }
  }

  private _handleMouse(e: MouseEvent) {
    const context = this._getContext()

    this._pointerInDataCoord.animation?.kill()
    this._pointerInDataCoord.transition = {
      from: interpolate(this._pointerInDataCoord.transition),
      target: _getPointerInDataCoord(
        fGet(localPoint(this._canvas, e)),
        this._getContext().derivedState.target,
      ),
      progress: 0,
    }
    this._pointerInDataCoord.animation = this._registerAnimation(
      gsap.to(this._pointerInDataCoord.transition, {
        progress: 1,
        duration: pointerAnimationDuration,
        ease: 'power4',
      }),
    )
    this._components.forEach(
      (x) => x.update?.('pointer', context, this._callbacks.registerAnimation),
    )
    this.draw()
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

// const _interpolateState = (
//   stateTransition: ChartDataTransition<ChartState>,
// ): ChartState => ({
//   xyRange: chartDataTransitionCurrObj(stateTransition, (x) => x.xyRange),
// })

export type ChartStateDerived = ReturnType<typeof _chartDerivedState>
const _chartDerivedState = (sizing: ChartSizing, state: ChartState) => {
  const { xyRange } = state
  const { size, padding } = sizing
  const viewport = rectExt({ x: 0, y: 0, ...size })
  const plotArea = rectExt({
    x: viewport.x + padding.left,
    y: viewport.y + padding.top,
    width: viewport.width - padding.left - padding.right,
    height: viewport.height - padding.top - padding.bottom,
  })
  const scale = {
    x: linearFnFomPoints(
      xyRange.x.start,
      plotArea.x,
      xyRange.x.end,
      plotArea.width + plotArea.x,
    ),
    y: linearFnFomPoints(
      xyRange.y.start,
      plotArea.y + plotArea.height,
      xyRange.y.end,
      plotArea.y,
    ),
  }
  return { plotArea, scale, viewport, padding }
}

function _getPointerInDataCoord(
  pointerInPixelsCoord: XY,
  { scale, plotArea }: ChartStateDerived,
) {
  return {
    x: _.clamp(
      Math.round(scale.x.inverse(pointerInPixelsCoord.x)),
      Math.round(scale.x.inverse(plotArea.x)),
      Math.round(scale.x.inverse(plotArea.right)),
    ),
    y: _.clamp(
      Math.round(scale.y.inverse(pointerInPixelsCoord.y)),
      Math.round(scale.y.inverse(plotArea.y)),
      Math.round(scale.y.inverse(plotArea.right)),
    ),
  }
}
