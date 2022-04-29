import {gsap} from 'gsap'
import _ from 'lodash'
import {Padding, RectExt, rectExt, Size} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {fGet} from '../../../Utils/Utils'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {ChartContext} from './ChartContext'
import {
  ChartDataTransition,
  chartDataTransitionCurrNum,
  chartDataTransitionCurrObj,
} from './ChartUtils/ChartDataTransition'

export type ChartXYRange = {x: SimpleRange; y: SimpleRange}
export type ChartAnimation = {ease: gsap.EaseFunction; duration: number}

export type ChartState = {
  area: (size: Size) => {viewport: RectExt; padding: Padding}
  xyRange: ChartXYRange
  alpha: number
}

export class Chart<Data> {
  private _canvasContext: CanvasRenderingContext2D
  private _pointer: {x: number; y: number}
  private _dataTransition: ChartDataTransition<Data>
  private _dataAnimation: ReturnType<typeof gsap.to> | null = null
  private _stateTransition: ChartDataTransition<ChartState>
  private _stateAnimation: ReturnType<typeof gsap.to> | null = null
  private _components: readonly ChartComponent<Data>[]

  constructor(
    canvas: HTMLCanvasElement,
    data: Data,
    state: ChartState,
    components: readonly ChartComponent<Data>[],
    private debugName: string,
    private _size: Size,
    private _registerAnimation: <
      T extends gsap.core.Tween | gsap.core.Timeline
    >(
      tween: T
    ) => T
  ) {
    this._components = components
    this._canvasContext = fGet(canvas.getContext('2d'))

    this._dataTransition = {target: data, prev: data, transition: 1}
    this._stateTransition = {target: state, prev: state, transition: 1}

    this._pointer = {x: this._size.width * 0.25, y: 0}
    const context = this._getContext()
    this._components.forEach(x =>
      x.update?.('init', context, this._registerAnimation)
    )
  }

  destroy() {
    this._components.forEach(x => x.destroy?.())
    this._dataAnimation?.kill()
    this._stateAnimation?.kill()
  }

  setComponents(components: readonly ChartComponent<Data>[]) {
    this._components.forEach(x => x.destroy?.())
    this._components = components
    const context = this._getContext()
    this._components.forEach(x =>
      x.update?.('init', context, this._registerAnimation)
    )
  }

  updatePointer(pointer: {x: number; y: number}) {
    this._pointer = pointer
    const context = this._getContext()
    this._components.forEach(x =>
      x.update?.('pointer', context, this._registerAnimation)
    )
  }

  updateState(data: Data, state: ChartState, animation: ChartAnimation | null) {
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
            {transition: 0},
            {transition: 1, ease: animation.ease, duration: animation.duration}
          )
        )
      } else {
        this._dataTransition.transition = 1
      }
    }

    this._stateTransition.prev = _interpolateState(this._stateTransition)
    this._stateTransition.target = state
    this._stateAnimation?.kill()
    if (animation) {
      this._stateAnimation = this._registerAnimation(
        gsap.fromTo(
          this._stateTransition,
          {transition: 0},
          {transition: 1, ease: animation.ease, duration: animation.duration}
        )
      )
    } else {
      this._stateTransition.transition = 1
    }

    const context = this._getContext()
    this._components.forEach(x =>
      x.update?.('state', context, this._registerAnimation)
    )
  }

  setSize(size: Size) {
    if (_.isEqual(size, this._size)) {
      return
    }
    this._size = size
    const context = this._getContext()
    this._components.forEach(x =>
      x.update?.('size', context, this._registerAnimation)
    )
  }

  draw() {
    const context = this._getContext()
    const {canvasContext} = context
    canvasContext.globalAlpha = context.currState.alpha
    this._components.forEach(component => {
      canvasContext.save()
      component.draw(context)
      canvasContext.restore()
    })
  }

  private _getContext(): ChartContext<Data> {
    const currState = _interpolateState(this._stateTransition)
    return {
      pointer: this._pointer,
      canvasContext: this._canvasContext,
      dataTransition: this._dataTransition,
      stateTransition: this._stateTransition,
      currState,
      derivedState: {
        prev: _chartDerivedState(this._size, this._stateTransition.prev),
        target: _chartDerivedState(this._size, this._stateTransition.target),
        curr: _chartDerivedState(this._size, currState),
      },
      chartSize: this._size,
    }
  }
}

const _interpolateState = (
  stateTransition: ChartDataTransition<ChartState>
): ChartState => {
  const {transition} = stateTransition
  const xyRange = chartDataTransitionCurrObj(stateTransition, x => x.xyRange)
  const alpha = chartDataTransitionCurrNum(stateTransition, x => x.alpha)
  const prevArea = stateTransition.prev.area
  const targetArea = stateTransition.target.area
  const area = (size: Size) =>
    chartDataTransitionCurrObj(
      {transition, prev: prevArea(size), target: targetArea(size)},
      x => x
    )
  return {
    xyRange,
    alpha,
    area,
  }
}

export type ChartStateDerived = ReturnType<typeof _chartDerivedState>
const _chartDerivedState = (size: Size, state: ChartState) => {
  const {xyRange, area} = state
  const {viewport, padding} = area(size)
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
  return {plotArea, scale, viewport, padding}
}
