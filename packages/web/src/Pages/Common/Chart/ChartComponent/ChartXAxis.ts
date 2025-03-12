import { assert, block, fGet, letIn, linearFnFomPoints } from '@tpaw/common'
import _ from 'lodash'
import { rectExt } from '../../../../Utils/Geometry'
import { interpolate } from '../../../../Utils/Interpolate'
import { Record } from '@tpaw/common'
import {
  transitionArrayDecompose,
  transitionTransform,
} from '../../../../Utils/Transition'
import { PlanColors } from '../../../PlanRoot/Plan/UsePlanColors'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export type ChartXAxisTickType = 'large' | 'medium' | 'small' | 'none'

export type ChartXAxisProps = {
  colors: PlanColors
  marginTop: number
  padding: { top: number }
  height: number
  markers: {
    start: number
    retirement: number
    end: number
  }
  label: {
    text: string
    fontSize: number
    paddingLeft: number
  }
  pointer: {
    height: number
  } | null
  transformDataXDelta: number
  getTick: (
    transformedDataX: number,
    pixelsPerTransformedDataX: number,
  ) => {
    color: string
    line: { length: number }
    label: {
      text: string
      paddingTop: number
      fontSize: number
      fontWeight: ChartUtils.FontWeight
      visible: boolean
    }
  } | null
}

export class ChartXAxis<Params> implements ChartComponent<Params> {
  constructor(
    private readonly _propsFn: (params: Params) => ChartXAxisProps | null,
  ) {}

  public draw(chartContext: ChartContext<Params>) {
    const { canvasContext: ctx, stateTransition, currState } = chartContext

    const fullTransition = transitionTransform(stateTransition, (x) => {
      const props = this._propsFn(x.params)
      return {
        ...x,
        props,
        transformedDataRange: props
          ? Record.mapValues(
              x.dataRange.x,
              (x) => props.transformDataXDelta + x,
            )
          : null,
      }
    })
    if (
      fullTransition.from.props === null &&
      fullTransition.target.props === null
    )
      return null

    const transformedDataXs = block(() => {
      const from = fullTransition.from.transformedDataRange
      const target = fullTransition.target.transformedDataRange
      const fFrom = from ?? fGet(target)
      const fTarget = target ?? fGet(from)
      const range = {
        start: Math.min(fFrom.start, fTarget.start),
        end: Math.max(fFrom.end, fTarget.end),
      }
      return _.range(range.start, range.end + 1)
    })

    const pixelTransition = transitionTransform(
      fullTransition,
      ({ derivedState, props, dataRange }) => {
        if (!props) return null
        const { plotArea, scale } = derivedState
        const transformDataX = (dataX: number) =>
          dataX + props.transformDataXDelta
        transformDataX.inverse = (transformedDataX: number) =>
          transformedDataX - props.transformDataXDelta

        const origin = {
          x: 0,
          y: plotArea.y + plotArea.height + props.marginTop,
        }

        const pixelPerDataX = scale.x.start(2) - scale.x.start(1)
        const bg = block(() => ({
          x: scale.x.start(dataRange.x.start),
          y: 0,
          right: scale.x.start(props.markers.end),
          height: props.height,
          retirement: {
            start: scale.x.start(props.markers.retirement),
          },
        }))

        type TickInfo = Exclude<ReturnType<typeof props.getTick>, null>
        const getPixelTick = (tick: TickInfo, x: number) => {
          const line = {
            y: props.padding.top,
            length: tick.line.length,
          }
          return {
            x,
            color: tick.color,
            line,
            label: {
              text: tick.label.text,
              y: line.y + line.length + tick.label.paddingTop,
              fontSize: tick.label.fontSize,
              fontWeight: tick.label.fontWeight,
              opacity: tick.label.visible ? 1 : 0,
            },
          }
        }
        return {
          interpolatable: {
            origin,
            bg,
            label: block(() => {
              const text = (
                fullTransition.target.props ?? fGet(fullTransition.from.props)
              ).label.text

              ctx.font = ChartUtils.getFont(props.label.fontSize, '600')
              ctx.textBaseline = 'top'
              const measure = ctx.measureText(text)
              return {
                x: bg.x + props.label.paddingLeft,
                width: measure.width,
                y: (props.height - measure.actualBoundingBoxDescent) / 2,
                text,
                fontSize: props.label.fontSize,
              }
            }),
            minMaxTicks: block(() => {
              const getNonNullTickAtOrBefore = (dataX: number): TickInfo =>
                props.getTick(transformDataX(dataX), pixelPerDataX) ??
                getNonNullTickAtOrBefore(dataX - 1)

              const minTick = getNonNullTickAtOrBefore(dataRange.x.start)

              return {
                min: {
                  ...getPixelTick(minTick, bg.x),
                  textAlign: 'left' as const,
                  opacity: 1,
                },
                max: block(() => {
                  const tick = getNonNullTickAtOrBefore(props.markers.end)
                  const pixelTick = getPixelTick(tick, bg.right - 1)
                  ctx.font = ChartUtils.getFont(
                    pixelTick.label.fontSize,
                    pixelTick.label.fontWeight,
                  )
                  const labelWidth = ctx.measureText(pixelTick.label.text).width
                  return {
                    ...pixelTick,
                    textAlign: 'right' as const,
                    opacity: tick.label.visible ? 0 : 1,
                    maxTickRight: tick.label.visible
                      ? bg.right - 1
                      : pixelTick.x - labelWidth,
                  }
                }),
              }
            }),
            pointer: block(() => {
              const x = currState.pointer.position.visual.x - origin.x
              const height = props.pointer?.height ?? 0
              const { y0, width } = interpolate({
                from: { y0: -height, width: height * 1.5 },
                target: { y0: 0, width: height * 2 },
                progress: currState.pointer.hover,
              })

              const opacity = props.pointer
                ? _.clamp(linearFnFomPoints(0, 1, 5, 0)(x - bg.right), 0, 1)
                : 0

              return {
                points: [
                  { x: x, y: y0 },
                  { x: x + width / 2, y: y0 + height },
                  { x: x - width / 2, y: y0 + height },
                ],
                opacity,
              }
            }),
          },
          ticks: transformedDataXs.map((transformedX) =>
            letIn(props.getTick(transformedX, pixelPerDataX), (tick) =>
              tick
                ? getPixelTick(
                    tick,
                    scale.x.start(transformDataX.inverse(transformedX)),
                  )
                : null,
            ),
          ),
        }
      },
    )

    const fPixelTransition = transitionTransform(
      pixelTransition,
      (x, other) => x ?? fGet(other),
    )

    const colors = (
      fullTransition.target.props ?? fGet(fullTransition.from.props)
    ).colors
    const { shades } = colors

    ctx.globalAlpha = interpolate(
      transitionTransform(fullTransition, (x) => (x.props ? 1 : 0)),
    )
    const { origin, bg, label, minMaxTicks, pointer } = interpolate(
      transitionTransform(fPixelTransition, (x) => x.interpolatable),
    )

    ctx.translate(origin.x, origin.y)

    ChartUtils.saveRestore(ctx, () => {
      // Set the clip as the bg.
      const bgPath = ChartUtils.roundRect(rectExt(bg), 2)
      ctx.clip(bgPath)

      // Draw the bg.
      ctx.fillStyle = shades.light[4].hex
      ctx.fill(bgPath)
      ctx.fillStyle = shades.light[10].hex
      ctx.fillRect(
        bg.retirement.start,
        bg.y,
        bg.right - bg.retirement.start,
        bg.height,
      )

      // Draw the label.
      ChartUtils.saveRestore(ctx, () => {
        ctx.font = ChartUtils.getFont(label.fontSize, '600')
        ctx.textBaseline = 'top'
        ctx.fillStyle = colors.dark.hex
        ctx.fillText(label.text, label.x, label.y)
      })

      // Draw the minMaxTicks
      ;[minMaxTicks.min, minMaxTicks.max].forEach((tick) =>
        ChartUtils.saveRestore(ctx, () => {
          ctx.globalAlpha = tick.opacity
          ctx.fillStyle = tick.color
          ctx.font = ChartUtils.getFont(
            tick.label.fontSize,
            tick.label.fontWeight,
          )
          ctx.textBaseline = 'top'
          ctx.textAlign = tick.textAlign
          ctx.fillText(tick.label.text, tick.x, tick.label.y)
        }),
      )

      // Draw the ticks.
      transitionArrayDecompose(
        transitionTransform(fPixelTransition, (x) => x.ticks),
      ).forEach(({ from, target, progress }) => {
        if (from === null) {
          assert(target === null)
          return
        }
        assert(target !== null)

        const curr = interpolate({ from, target, progress })

        // Draw the line.
        if (curr.x > bg.x + 2 && curr.x < bg.right - 2) {
          ctx.strokeStyle = curr.color
          ctx.lineWidth = 1
          ctx.stroke(
            block(() => {
              const linePath = new Path2D()
              linePath.moveTo(curr.x, curr.line.y)
              linePath.lineTo(curr.x, curr.line.y + curr.line.length)
              return linePath
            }),
          )
        }

        // Draw the label.
        if (
          curr.label.opacity !== 0 &&
          curr.x > label.x + label.width + 15 &&
          curr.x <= bg.right
        ) {
          ChartUtils.saveRestore(ctx, () => {
            ctx.globalAlpha = curr.label.opacity
            ctx.fillStyle = curr.color
            ctx.font = ChartUtils.getFont(
              curr.label.fontSize,
              curr.label.fontWeight,
            )
            ctx.textBaseline = 'top'
            ctx.textAlign = 'center'
            ctx.fillText(
              curr.label.text,
              Math.min(
                curr.x,
                minMaxTicks.max.maxTickRight -
                  ctx.measureText(curr.label.text).width / 2,
              ),
              curr.label.y,
            )
          })
        }
      })
    })
    // Draw the pointer.
    ChartUtils.saveRestore(ctx, () => {
      ctx.clip(
        block(() => {
          const path = new Path2D()
          path.rect(
            currState.derivedState.viewport.x,
            0,
            currState.derivedState.viewport.width,
            10000,
          )
          return path
        }),
      )
      ctx.globalAlpha = pointer.opacity
      ctx.fillStyle = colors.dark.hex
      ctx.fill(ChartUtils.roundedPolygon(pointer.points, 2))
    })
    return null
  }
}
