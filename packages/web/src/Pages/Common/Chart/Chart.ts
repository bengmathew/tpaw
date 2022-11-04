import { linearFnFomPoints } from '@tpaw/common'
import localPoint from '@visx/event/lib/localPoint'
import { gsap } from 'gsap'
import _ from 'lodash'
import { Padding, rectExt, Size } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { assert, fGet } from '../../../Utils/Utils'
import { ChartComponent } from './ChartComponent/ChartComponent'
import { ChartContext } from './ChartContext'
import {
  ChartDataTransition,
  chartDataTransitionCurrObj,
} from './ChartUtils/ChartDataTransition'

export type ChartXYRange = { x: SimpleRange; y: SimpleRange }
export type ChartAnimation = { ease: gsap.EaseFunction; duration: number }
export type ChartState = { xyRange: ChartXYRange }
export type ChartSizing = { size: Size; padding: Padding }

export class Chart<Data> {
  private _pointer: { x: number; y: number }
  private _dataTransition: ChartDataTransition<Data>
  private _dataAnimation: ReturnType<typeof gsap.to> | null = null
  private _stateTransition: ChartDataTransition<ChartState>
  private _stateAnimation: ReturnType<typeof gsap.to> | null = null
  private _components: readonly ChartComponent<Data>[]

  private _callbacks = {
    draw: this.draw.bind(this),
    registerAnimation: this._registerAnimation.bind(this),
    onMouse: this._handleMouse.bind(this),
  }
  private _activeAnimations: (gsap.core.Tween | gsap.core.Timeline)[] = []

  constructor(
    private _canvas: HTMLCanvasElement,
    data: Data,
    xyRange: ChartXYRange,
    components: readonly ChartComponent<Data>[],
    private _sizing: ChartSizing,
  ) {
    this._components = components

    this._dataTransition = { target: data, prev: data, transition: 1 }
    this._stateTransition = {
      target: { xyRange },
      prev: { xyRange },
      transition: 1,
    }

    this._pointer = { x: this._sizing.size.width * 0.25, y: 0 }
    const context = this._getContext()
    this._components.forEach((x) =>
      x.update?.('init', context, this._callbacks.registerAnimation),
    )
    _canvas.addEventListener('pointermove', this._callbacks.onMouse)
    _canvas.addEventListener('pointerenter', this._callbacks.onMouse)
    _canvas.addEventListener('pointerleave', this._callbacks.onMouse)
  }

  destroy() {
    this._components.forEach((x) => x.destroy?.())
    this._dataAnimation?.kill()
    this._stateAnimation?.kill()
    assert(this._activeAnimations.length === 0)
    this._canvas.removeEventListener('pointermove', this._callbacks.onMouse)
    this._canvas.removeEventListener('pointerenter', this._callbacks.onMouse)
    this._canvas.removeEventListener('pointerleave', this._callbacks.onMouse)
  }

  setComponents(components: readonly ChartComponent<Data>[]) {
    this._components.forEach((x) => x.destroy?.())
    this._components = components
    const context = this._getContext()
    this._components.forEach((x) =>
      x.update?.('init', context, this._callbacks.registerAnimation),
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
      this._dataTransition.prev = this._dataTransition.target
      this._dataTransition.target = data
      this._dataAnimation?.kill()
      this._dataAnimation = null
      if (animation) {
        this._dataAnimation = this._registerAnimation(
          gsap.fromTo(
            this._dataTransition,
            { transition: 0 },
            {
              transition: 1,
              ease: animation.ease,
              duration: animation.duration,
            },
          ),
        )
      } else {
        this._dataTransition.transition = 1
      }
    }

    this._stateTransition.prev = _interpolateState(this._stateTransition)
    this._stateTransition.target = { xyRange }
    this._stateAnimation?.kill()
    if (animation) {
      this._stateAnimation = this._registerAnimation(
        gsap.fromTo(
          this._stateTransition,
          { transition: 0 },
          { transition: 1, ease: animation.ease, duration: animation.duration },
        ),
      )
    } else {
      this._stateTransition.transition = 1
    }

    const context = this._getContext()
    this._components.forEach((x) =>
      x.update?.('state', context, this._callbacks.registerAnimation),
    )
    this.draw()
  }

  setSizing(sizing: ChartSizing) {
    if (_.isEqual(sizing, this._sizing)) return

    this._sizing = sizing
    const context = this._getContext()
    this._components.forEach((x) =>
      x.update?.('sizing', context, this._callbacks.registerAnimation),
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
    const currState = _interpolateState(this._stateTransition)
    return {
      pointer: this._pointer,
      canvasContext: fGet(this._canvas.getContext('2d')),
      dataTransition: this._dataTransition,
      stateTransition: this._stateTransition,
      currState,
      derivedState: {
        prev: _chartDerivedState(this._sizing, this._stateTransition.prev),
        target: _chartDerivedState(this._sizing, this._stateTransition.target),
        curr: _chartDerivedState(this._sizing, currState),
      },
      sizing: this._sizing,
    }
  }

  private _handleMouse(e: MouseEvent) {
    this._pointer = fGet(localPoint(this._canvas, e))
    const context = this._getContext()
    this._components.forEach((x) =>
      x.update?.('pointer', context, this._callbacks.registerAnimation),
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

const _interpolateState = (
  stateTransition: ChartDataTransition<ChartState>,
): ChartState => ({
  xyRange: chartDataTransitionCurrObj(stateTransition, (x) => x.xyRange),
})

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
