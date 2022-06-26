import {gsap} from 'gsap'
import _ from 'lodash'
import {assert, fGet, noCase} from '../../../../Utils/Utils'
import {ChartState, ChartStateDerived} from '../Chart'
import {ChartContext} from '../ChartContext'
import {
  ChartDataTransition,
  chartDataTransitionCurrObj,
  chartDataTransitionTransform,
} from '../ChartUtils/ChartDataTransition'
import {zeroOneInterpolate} from '../ChartUtils/ZeroOneInterpolate'
import {ChartComponent, ChartRegisterAnimation} from './ChartComponent'
import {chartPointerBox, ChartPointerBoxAnimatedState} from './ChartPointerBox'
import {
  chartPointerXAxis,
  ChartPointerXAxisAnimatedState,
} from './ChartPointerXAxis'

const duration = 1

type _DataFn<Data> = (
  data: Data
) => ({line: (x: number) => number; label: string} | null)[]

export type ChartPointerState = {dataX: number}
export type ChartPointerContext<Data> = ChartContext<Data> & {
  pointerTransition: ChartDataTransition<ChartPointerState>
}

export type ChartPointerComponentTargetArgs = {
  dataX: number
  dataYInfos: {dataY: number; label: string}[]
  chartState: ChartState
  chartStateDerived: ChartStateDerived

  // These can be used in the draw method.
  ctx: CanvasRenderingContext2D
  displayX: (x: number) => number
  formatY: (y: number) => string
  formatX: (x: number, type: 'long' | 'short') => string
  markDataX: number
}

export type ChartPointerComponentDrawArg<AnimatedProps> = {
  dataX: number
  dataYInfos: {dataY: number; label: string}[]
  chartState: ChartState
  chartStateDerived: ChartStateDerived
  animatedProps: AnimatedProps
  // Other values remain constant between target and draw.
}

export type ChartPointerComponent<AnimatedProps> = (
  args: ChartPointerComponentTargetArgs
) => {
  animated: AnimatedProps
  draw: (args: ChartPointerComponentDrawArg<AnimatedProps>) => void
}

export class ChartPointer<Data> implements ChartComponent<Data> {
  private _state: {
    transition: ChartDataTransition<{
      dataX: number
      box: ChartPointerBoxAnimatedState
      xAxis: ChartPointerXAxisAnimatedState
    }>
    draws: {
      box: (
        args: ChartPointerComponentDrawArg<ChartPointerBoxAnimatedState>
      ) => void
      xAxis: (
        args: ChartPointerComponentDrawArg<ChartPointerXAxisAnimatedState>
      ) => void
    }
  } | null = null
  private _animation: gsap.core.Tween | null = null

  constructor(
    private _dataFn: _DataFn<Data>,
    private _displayX: (data: Data, x: number) => number,
    private _formatX: (data: Data, x: number, type: 'long' | 'short') => string,
    private _formatY: (data: Data, x: number) => string,
    private _markFn: (data: Data) => number
  ) {}

  destroy() {
    this._animation?.kill()
  }

  update(
    change: 'init' | 'pointer' | 'state' | 'sizing',
    context: ChartContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) {
    const {
      canvasContext: ctx,
      pointer,
      derivedState,
      dataTransition,
      stateTransition,
    } = context
    const dataX = (() => {
      switch (change) {
        case 'init':
        case 'pointer': {
          const {scale, plotArea} = derivedState.target
          return Math.round(
            _.clamp(
              scale.x.inverse(pointer.x),
              scale.x.inverse(plotArea.x),
              scale.x.inverse(plotArea.right)
            )
          )
        }
        case 'state': {
          assert(this._state)
          const {scale, plotArea} = derivedState.target
          return _.clamp(
            this._state.transition.target.dataX,
            Math.ceil(scale.x.inverse(plotArea.x)),
            Math.floor(scale.x.inverse(plotArea.right))
          )
        }

        case 'sizing':
          assert(this._state)
          return this._state.transition.target.dataX
        default:
          noCase(change)
      }
    })()

    const targetArgs: ChartPointerComponentTargetArgs = {
      ctx,
      dataX,
      dataYInfos: _.compact(
        _.reverse(
          this._dataFn(dataTransition.target).map(x =>
            x === null
              ? null
              : {
                  dataY: x.line(dataX),
                  label: x.label,
                }
          )
        )
      ),
      chartState: stateTransition.target,
      chartStateDerived: derivedState.target,
      displayX: x => this._displayX(dataTransition.target, x),
      formatY: y => this._formatY(dataTransition.target, y),
      formatX: (x, type) => this._formatX(dataTransition.target, x, type),
      markDataX: Math.round(this._markFn(dataTransition.target)),
    }
    ctx.save()
    const boxInfo = chartPointerBox(targetArgs)
    ctx.restore()
    ctx.save()
    const xAxisInfo = chartPointerXAxis(targetArgs)
    ctx.restore()

    const target = {dataX, box: boxInfo.animated, xAxis: xAxisInfo.animated}
    if (change === 'init') {
      assert(!this._state)
      this._state = {
        draws: {box: boxInfo.draw, xAxis: xAxisInfo.draw},
        transition: {prev: target, target, transition: 1},
      }
    } else {
      assert(this._state)
      const curr = chartDataTransitionCurrObj(this._state.transition, x => x)

      // Take care not to maintain the same transition object so animation
      // will not reference old object.
      this._state = {
        draws: {box: boxInfo.draw, xAxis: xAxisInfo.draw},
        transition: this._state.transition,
      }

      this._state.transition.target = target

      switch (change) {
        case 'sizing':
          break
        case 'pointer':
        case 'state':
          this._state.transition.prev = curr
          this._state.transition.transition = 0
          this._animation?.kill()
          this._animation = registerAnimation(
            gsap.to(this._state.transition, {
              transition: 1,
              duration,
              ease: 'power4',
            })
          )
          break
        default:
          noCase(change)
      }
    }
  }

  draw(context: ChartContext<Data>) {
    const {
      canvasContext: ctx,
      dataTransition,
      derivedState,
      currState,
    } = context

    assert(this._state)
    const currTransition = chartDataTransitionCurrObj(
      this._state.transition,
      x => x
    )
    const {dataX} = currTransition
    const lineTransition = chartDataTransitionTransform(dataTransition, x =>
      this._dataFn(x).map(x => x?.line(dataX) ?? null)
    )
    const labels = this._dataFn(dataTransition.target).map(
      x => x?.label ?? null
    )

    const dataYInfos = _.compact(
      _.reverse(
        lineTransition.target.map((target, i) => {
          const prev = lineTransition.prev[i]
          if (target === null) return null
          if (prev === null) return target
          return zeroOneInterpolate(prev, target, lineTransition)
        })
      ).map((dataY, i) =>
        dataY === null ? null : {dataY, label: fGet(labels[i])}
      )
    )

    const drawArgs = <A>(animatedProps: A) => ({
      dataX,
      dataYInfos,
      chartState: currState,
      chartStateDerived: derivedState.curr,
      animatedProps,
    })

    // Draw the box.
    ctx.save()
    this._state.draws.box(drawArgs(currTransition.box))
    ctx.restore()

    // Draw the xAxis
    ctx.save()
    this._state.draws.xAxis(drawArgs(currTransition.xAxis))
    ctx.restore()
  }
}
