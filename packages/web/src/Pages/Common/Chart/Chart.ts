import {localPoint} from '@visx/event'
import {gsap} from 'gsap'
import _ from 'lodash'
import {LinearFn, linearFnFomPoints} from '../../../Utils/LinearFn'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {assert, fGet} from '../../../Utils/Utils'
import {ChartComponent} from './ChartComponent/ChartComponent'
import {ChartContext} from './ChartContext'
import {
  ChartDataTransition,
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
const duration = 1

type _ChartStateBase<Data> = {
  size: {width: number; height: number}
  padding: ChartPadding
  data: Data
  xyRange: ChartXYRange
}
type _ChartStateDerived = {
  viewPort: RectExt
  plotArea: RectExt
  scale: {x: LinearFn; y: LinearFn}
}
export type ChartState<Data> = _ChartStateBase<Data> & _ChartStateDerived

export class Chart<Data> {
  private _canvas
  private _ctx: CanvasRenderingContext2D
  private readonly _handleDraw = () => this._draw()
  private _id = _.uniqueId()

  private _stateKey: number | string = 0
  private _stateTransition: ChartDataTransition<ChartState<Data>>
  private _components: readonly ChartComponent<Data>[]
  private _stateAnimation: ReturnType<typeof gsap.to> | null = null
  private _activeAnimations: (gsap.core.Tween | gsap.core.Timeline)[] = []

  private get _registerAnimation() {
    return <T extends gsap.core.Tween | gsap.core.Timeline>(tween: T): T => {
      tween.eventCallback('onStart', () => {
        this._activeAnimations.push(tween)
        if (this._activeAnimations.length === 1) {
          gsap.ticker.add(this._handleDraw)
        }
      })
      const handleDone = () => {
        _.remove(this._activeAnimations, x => x === tween)
        if (this._activeAnimations.length === 0) {
          gsap.ticker.remove(this._handleDraw)
        }
      }
      tween.eventCallback('onComplete', handleDone)
      // As per https://greensock.com/forums/topic/22563-working-example-of-oninterrupt-callback/
      // this obviates the need to handle "kill"
      tween.eventCallback('onInterrupt', handleDone)
      return tween
    }
  }

  constructor(
    canvas: HTMLCanvasElement,
    baseState: _ChartStateBase<Data>,
    stateKey: number | string,
    components: readonly ChartComponent<Data>[]
  ) {
    this._canvas = canvas
    this._components = components

    const handlePointerPosition = (position: {x: number; y: number} | null) => {
      this._components.forEach(
        x =>
          x.setPointerPosition &&
          x.setPointerPosition(
            position,
            this._stateTransition.target,
            this._ctx,
            this._registerAnimation
          )
      )
      this._draw()
    }
    this._canvas.onpointermove = e =>
      handlePointerPosition(fGet(localPoint(this._canvas, e)))

    this._canvas.onpointerenter = e =>
      handlePointerPosition(fGet(localPoint(this._canvas, e)))

    this._canvas.onpointerleave = () => handlePointerPosition(null)

    this._ctx = fGet(canvas.getContext('2d'))

    const state = {...baseState, ..._derivedState(baseState)}
    this._stateKey = stateKey
    this._stateTransition = {target: state, prev: state, transition: 1}
    this.setState(baseState, stateKey)
    gsap.ticker.add(this._handleDraw)
  }

  destroy() {
    this._components.forEach(x => {
      if (x.destroy) x.destroy()
    })
    this._stateAnimation?.kill()
    assert(this._activeAnimations.length === 0)
  }

  setComponents(components: readonly ChartComponent<Data>[]) {
    this._components.forEach(x => {
      if (x.destroy) x.destroy()
    })
    this._components = components
    this._draw()
  }
  setState(baseState: _ChartStateBase<Data>, stateKey: string | number) {
    const {size, padding, data, xyRange} = baseState
    const target = {
      size,
      padding,
      data,
      xyRange,
      ..._derivedState(baseState),
    }
    this._stateTransition.prev =
      this._stateKey === stateKey
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
    this._stateKey = stateKey
    this._stateAnimation?.kill()
    this._stateAnimation = this._registerAnimation(
      gsap.fromTo(
        this._stateTransition,
        {transition: 0},
        {transition: 1, ease: 'power4', duration}
      )
    )
    if (
      size.width !== this._stateTransition.prev.size.width ||
      size.height !== this._stateTransition.prev.size.width
    ) {
      const canvas = this._canvas
      canvas.style.width = `${size.width}px`
      canvas.style.height = `${size.height}px`
      const dpr = window.devicePixelRatio || 1
      canvas.width = size.width * dpr
      canvas.height = size.height * dpr
      this._ctx.scale(dpr, dpr)
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
    this._draw()
  }
  private _draw() {
    const ctx = this._ctx
    const stateTransition = this._stateTransition
    const currState = _interpolateState(stateTransition)
    const chartContext: ChartContext<Data> = {ctx, stateTransition, currState}

    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height)

    this._components.forEach(component => {
      ctx.save()
      component.draw(chartContext, this._registerAnimation)
      ctx.restore()
    })
  }
}

const _derivedState = ({
  size,
  padding,
  xyRange,
}: {
  size: {width: number; height: number}
  padding: ChartPadding
  xyRange: ChartXYRange
}): _ChartStateDerived => {
  const viewPort = rectExt({x: 0, y: 0, ...size})
  const plotArea = rectExt({
    x: padding.left,
    y: padding.top,
    width: viewPort.width - padding.left - padding.right,
    height: viewPort.height - padding.top - padding.bottom,
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
  return {
    viewPort,
    plotArea,
    scale,
  }
}

const _interpolateState = <Data>(
  stateTransition: ChartDataTransition<ChartState<Data>>
): Omit<ChartState<Data>, 'data'> => {
  // Don't interpolate the size.
  const size = stateTransition.target.size
  const padding = chartDataTransitionCurrObj(stateTransition, x => x.padding)
  const xyRange = chartDataTransitionCurrObj(stateTransition, x => x.xyRange)
  return {
    size,
    padding,
    xyRange,
    ..._derivedState({size, padding, xyRange}),
  }
}
