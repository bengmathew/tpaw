import {gsap} from 'gsap'
import {LinearFn, linearFnFomPoints} from '../../../Utils/LinearFn'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {fGet} from '../../../Utils/Utils'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {ChartContext} from './ChartContext'
import {
  ChartDataTransition,
  chartDataTransitionCurrNum,
  chartDataTransitionCurrObj,
} from './ChartUtils/ChartDataTransition'
import {rectExt, RectExt} from './ChartUtils/ChartUtils'

export type ChartXYRange = {x: SimpleRange; y: SimpleRange}
export type ChartPadding = {
  left: number
  right: number
  top: number
  bottom: number
}
export type ChartAnimation = {ease: gsap.EaseFunction; duration: number}

export type ChartState<Data> = {
  viewport: RectExt
  padding: ChartPadding
  data: Data
  xyRange: ChartXYRange
  alpha: number
}
type _ChartStateDerived = {
  plotArea: RectExt
  scale: {x: LinearFn; y: LinearFn}
}
export type ChartFullState<Data> = ChartState<Data> & _ChartStateDerived

export class Chart<Data> {
  private _canvas
  private _ctx: CanvasRenderingContext2D

  private _stateTransition: ChartDataTransition<ChartFullState<Data>>
  private _components: readonly ChartComponent<Data>[]
  private _stateAnimation: ReturnType<typeof gsap.to> | null = null

  pointerMoved(position: {x: number; y: number} | null) {
    const {viewport} = this._stateTransition.target

    if (
      position &&
      (position.x < viewport.x ||
        viewport.right <= position.x ||
        position.y < viewport.y ||
        viewport.bottom <= position.y)
    ) {
      position = null
    }
    this._components.forEach(x =>
      x.setPointerPosition?.(
        position,
        this._stateTransition.target,
        this._ctx,
        this._registerAnimation
      )
    )
    this.onDraw()
  }

  // private _handlers = {
  //   pointerMove: (e: PointerEvent) =>
  //     this._handlePointerPosition(fGet(localPoint(this._canvas, e))),
  //   pointerEnter: (e: PointerEvent) =>
  //     this._handlePointerPosition(fGet(localPoint(this._canvas, e))),
  //   pointerLeave: () => this._handlePointerPosition(null),
  // }

  constructor(
    canvas: HTMLCanvasElement,
    baseState: ChartState<Data>,
    components: readonly ChartComponent<Data>[],
    private debugName: string,
    private onDraw: () => void,
    private _registerAnimation: <
      T extends gsap.core.Tween | gsap.core.Timeline
    >(
      tween: T
    ) => T
  ) {
    this._canvas = canvas
    this._components = components
    this._ctx = fGet(canvas.getContext('2d'))

    const state = {...baseState, ..._derivedState(baseState)}
    this._stateTransition = {target: state, prev: state, transition: 1}
    this.setState(baseState, null)
  }

  destroy() {
    this._components.forEach(x => x.destroy?.())
    this._stateAnimation?.kill()
  }

  setComponents(components: readonly ChartComponent<Data>[]) {
    this._components.forEach(x => {
      if (x.destroy) x.destroy()
    })
    this._components = components
    this.onDraw()
  }
  setState(baseState: ChartState<Data>, animation: ChartAnimation | null) {
    const {viewport, padding, data, xyRange, alpha} = baseState
    const target = {
      viewport,
      padding,
      data,
      xyRange,
      alpha,
      ..._derivedState(baseState),
    }
    this._stateTransition.prev = animation
      ? {
          data:
            // This is a hack because we can't interpolate data. The correct
            // solution is to have a separate data transition that we accept
            // will be jerky if data changes before transition end, which is
            // probably not a common case. The rest of the state can change
            // without causing the jerk. This was encountered when padding
            // changed right after data changed.
            this._stateTransition.transition < 0.5 &&
            this._stateTransition.target.data === data
              ? this._stateTransition.prev.data
              : this._stateTransition.target.data,
          ..._interpolateState(this._stateTransition),
        }
      : target
    this._stateTransition.target = target

    this._stateAnimation?.kill()
    if (animation) {
      this._stateAnimation = this._registerAnimation(
        gsap.fromTo(
          this._stateTransition,
          {transition: 0},
          {transition: 1, ease: 'power4', duration: animation.duration}
        )
      )
    } else {
      this._stateTransition.transition = 1
    }

    this._components.forEach(
      x =>
        x.setState &&
        x.setState(
          this._stateTransition.target,
          this._ctx,
          this._registerAnimation
        )
    )
    this.onDraw()
  }
  draw() {
    const ctx = this._ctx
    const stateTransition = this._stateTransition
    const currState = _interpolateState(stateTransition)
    const chartContext: ChartContext<Data> = {ctx, stateTransition, currState}
    const {viewport, alpha} = currState

    ctx.globalAlpha = alpha
    this._components.forEach(component => {
      ctx.save()
      component.draw(chartContext, this._registerAnimation)
      ctx.restore()
    })
  }
}

const _derivedState = ({
  viewport,
  padding,
  xyRange,
}: {
  viewport: RectExt
  padding: ChartPadding
  xyRange: ChartXYRange
}): _ChartStateDerived => {
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
      plotArea.width + plotArea.x
    ),
    y: linearFnFomPoints(
      xyRange.y.start,
      plotArea.y + plotArea.height,
      xyRange.y.end,
      plotArea.y
    ),
  }
  return {plotArea, scale}
}

const _interpolateState = <Data>(
  stateTransition: ChartDataTransition<ChartFullState<Data>>
): Omit<ChartFullState<Data>, 'data'> => {
  const viewport = chartDataTransitionCurrObj(stateTransition, x => x.viewport)
  const padding = chartDataTransitionCurrObj(stateTransition, x => x.padding)
  const xyRange = chartDataTransitionCurrObj(stateTransition, x => x.xyRange)
  const alpha = chartDataTransitionCurrNum(stateTransition, x => x.alpha)
  return {
    viewport,
    padding,
    xyRange,
    alpha,
    ..._derivedState({viewport, padding, xyRange}),
  }
}
