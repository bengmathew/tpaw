import { SimpleRange, assert, block, fGet, letIn } from '@tpaw/common'
import { Power1, gsap } from 'gsap'
import _ from 'lodash'
import { interpolate } from '../../../../Utils/Interpolate'
import { Transition, transitionTransform } from '../../../../Utils/Transition'
import { ChartAnimatedProperty, ChartAnimation } from '../Chart'
import { ChartContext } from '../ChartContext'
import { ChartStyling } from '../ChartUtils/ChartStyling'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent, ChartRegisterAnimation } from './ChartComponent'
import { chartMergeDataComponentsArrTransition } from './ChartMergeDataComponentsArrTransition'

const _partHoverAnimation: ChartAnimation = {
  ease: Power1.easeOut,
  duration: 0.5,
}
const _getAnimation = (transition: Transition<0 | 1>) =>
  gsap.fromTo(
    transition,
    { progress: 0 },
    {
      progress: 1,
      ease: _partHoverAnimation.ease,
      duration: _partHoverAnimation.duration,
    },
  )

type ChartBreakdownProps = {
  total: {
    yByX: (number | null)[] | Float64Array
    xRange: SimpleRange | null
  }
  parts: {
    id: string
    data: {
      // Not null inside range (interpreted as closed interval).
      yByX: (number | null)[] | Float64Array
      xRange: SimpleRange | null // null is empty interval.
    }
    style: {
      fillPattern: ChartStyling.StripePattern
      pointer: {
        hover: ChartStyling.Stroke
        notHover: ChartStyling.Stroke
      }
    }
  }[]
  style: {
    total: {
      fillPattern: ChartStyling.StripePattern
      stroke: ChartStyling.Stroke
      pointer: {
        hover: ChartStyling.Stroke
        notHover: ChartStyling.Stroke
      }
    }
  }
}

const _deNullProps = (
  src: ChartBreakdownProps | null,
  other: ChartBreakdownProps | null,
  mergedParts: {
    id: string
    src: ChartBreakdownProps['parts'][number] | null
    other: ChartBreakdownProps['parts'][number] | null
  }[],
) => {
  return {
    total: {
      getY: (dataX: number) => (src ? _getYRangeChecked(dataX, src.total) : 0),
    },
    parts: mergedParts.map((part) => ({
      getY: (dataX: number) =>
        part.src ? _getYRangeChecked(dataX, part.src.data) : 0,
    })),
    style: {
      total: {
        fillPattern: fGet(src ?? other).style.total.fillPattern,
        stroke: ChartStyling.Stroke.deNull(
          src?.style.total.stroke ?? null,
          other?.style.total.stroke ?? null,
        ),
        pointer: {
          hover: ChartStyling.Stroke.deNull(
            src?.style.total.pointer.hover ?? null,
            other?.style.total.pointer.hover ?? null,
            { fixedWidth: true },
          ),
          notHover: ChartStyling.Stroke.deNull(
            src?.style.total.pointer.notHover ?? null,
            other?.style.total.pointer.notHover ?? null,
            { fixedWidth: true },
          ),
        },
      },
      parts: mergedParts.map((part) => ({
        id: part.id,
        fillPattern: fGet(part.src ?? part.other).style.fillPattern,
        pointer: fGet(part.src ?? part.other).style.pointer,
      })),
    },
  }
}

export class ChartBreakdown<Params> implements ChartComponent<Params> {
  private _partHoverAnimationMap: Map<string, ChartAnimatedProperty<0 | 1>> =
    new Map()

  private _stripePatternPathFn: ChartStyling.StripePattern.PatternPathFn | null =
    null
  constructor(
    private _id: string,
    private _propsFn: (params: Params) => ChartBreakdownProps | null,
  ) {}

  destroy() {
    this._partHoverAnimationMap.forEach((x) => {
      x.animation?.kill()
    })
  }

  draw(
    context: ChartContext<Params>,
    change:
      | 'draw'
      | 'init'
      | 'pointer'
      | 'stateAndPointer'
      | 'sizingAndPointer',
    registerAnimation: ChartRegisterAnimation,
  ) {
    const { canvasContext: ctx, stateTransition, currState } = context
    const { derivedState } = currState
    const { plotArea, viewport } = derivedState

    const directTransition = transitionTransform(stateTransition, (x) => {
      const props = this._propsFn(x.params)
      const dataXRange = props
        ? [props.total.xRange, ...props.parts.map((x) => x.data.xRange)].reduce(
            SimpleRange.union,
          )
        : null
      return { ...x, props, dataXRange }
    })

    const dataXRange = SimpleRange.union(
      directTransition.from.dataXRange,
      directTransition.target.dataXRange,
    )
    if (!dataXRange) return null
    const dataXs = SimpleRange.Closed.toArr(dataXRange)

    const fullTransition = block(() => {
      const mergedParts = chartMergeDataComponentsArrTransition(
        directTransition.from.props?.parts ?? [],
        directTransition.target.props?.parts ?? [],
      )
      return transitionTransform(directTransition, (x, other, src) => {
        const adjMerged = mergedParts.map((x) => ({
          id: x.from ? x.from.id : x.target.id,
          src: x[src],
          other: src === 'from' ? x.target : x.from,
        }))
        return {
          ...x,
          props: _deNullProps(x.props, other.props, adjMerged),
        }
      })
    })

    const pixelTransition = transitionTransform(
      fullTransition,
      ({ derivedState: { scale }, props }, other, src) => ({
        y0: scale.y(0),
        pixelBands: dataXs.map((dataX) => {
          let currY = scale.y(0)
          return {
            x: scale.x.start(dataX),
            ys: [
              ...props.parts.map((part) => {
                const start = currY
                const height = scale.y(0) - scale.y(part.getY(dataX))
                const end = start - height
                currY = end
                return end
              }),
              scale.y(props.total.getY(dataX)),
            ],
          }
        }),
        style: {
          ...props.style,
          parts: [
            ...props.style.parts.map((x) => ({
              ...x,
              dataId: this.ids.part(x.id),
            })),
            {
              dataId: this.ids.remaining,
              fillPattern: props.style.total.fillPattern,
              pointer: props.style.total.pointer,
            },
          ],
        },
      }),
    )

    const { pixelBands, y0, style } = interpolate(pixelTransition)

    const pointerTargetYRanges = (dataX: number) => {
      const dataXOffset = dataX - dataXs[0]
      if (dataXOffset < 0 || dataXOffset > pixelBands.length - 1) return []
      const floor = Math.floor(dataXOffset)
      const ceil = Math.ceil(dataXOffset)

      let currEnd = null as Transition<number> | null
      return style.parts.map(({ dataId }, i) => {
        const start =
          currEnd === null
            ? transitionTransform(
                fullTransition,
                ({ derivedState: { scale } }) => scale.y(0),
              )
            : currEnd
        const end = transitionTransform(pixelTransition, ({ pixelBands }) =>
          floor === ceil
            ? pixelBands[floor].ys[i]
            : interpolate({
                from: pixelBands[floor].ys[i],
                target: pixelBands[ceil].ys[i],
                progress: (dataXOffset - floor) / (ceil - floor),
              }),
        )
        currEnd = end
        const mid = {
          from: start.from + (end.from - start.from) / 2,
          target: start.target + (end.target - start.target) / 2,
        }
        return {
          id: dataId,
          pixelYTransition: { start, end, mid },
        }
      })
    }

    if (change !== 'draw') {
      this._stripePatternPathFn?.destroy()
      this._stripePatternPathFn = ChartStyling.StripePattern.getPatternPathFn(
        context.stateTransition.target.derivedState.viewport,
      )
      const hoverPartId = block(() => {
        const { derivedState } = context.stateTransition.target
        const { scale } = derivedState
        const pointerPixelCoordinates =
          context.pointerTransition.visualPosition.target
        const pointerDataX = scale.x.inverse.rounded(pointerPixelCoordinates.x)
        return (
          pointerTargetYRanges(pointerDataX).find(
            ({ pixelYTransition: { start, end } }) =>
              start.target > end.target &&
              // Not <= as a simple hack to avoid being in target when
              // pointer is clamped to bounds.
              pointerPixelCoordinates.y < start.target &&
              pointerPixelCoordinates.y > end.target,
          )?.id ?? null
        )
      })

      // Cleanup.
      ;[...this._partHoverAnimationMap.entries()].forEach(([id, value]) => {
        assert(value.animation)
        if (value.transition.progress === 1 && value.transition.target === 0) {
          value.animation.kill()
          this._partHoverAnimationMap.delete(id)
        }
      })

      // Remove existing animation.
      const existingAnimation = !hoverPartId
        ? null
        : block(() => {
            const result = this._partHoverAnimationMap.get(hoverPartId) ?? null
            this._partHoverAnimationMap.delete(hoverPartId)
            return result
          })

      // Animate everything else to 0.
      ;[...this._partHoverAnimationMap.entries()].forEach(([id, value]) => {
        if (value.transition.target === 0) return
        assert(value.animation)
        value.animation.kill()
        value.transition = {
          from: interpolate(value.transition),
          target: 0,
          progress: 0,
        }
        value.animation = registerAnimation(_getAnimation(value.transition))
      })

      if (hoverPartId) {
        this._partHoverAnimationMap.set(
          hoverPartId,
          block(() => {
            if (existingAnimation && existingAnimation.transition.target === 1)
              return existingAnimation
            existingAnimation?.animation?.kill()
            const transition = {
              from: existingAnimation
                ? interpolate(existingAnimation.transition)
                : 0,
              target: 1 as const,
              progress: 0,
            }
            return {
              transition,
              animation: registerAnimation(_getAnimation(transition)),
            }
          }),
        )
      }
    }

    ChartUtils.saveRestore(ctx, () => {
      // Apply the clip.
      ctx.clip(
        block(() => {
          const horzBleed = Math.max(
            style.total.stroke.width,
            style.total.stroke.width,
            style.total.stroke.width,
          )
          const path = new Path2D()
          path.rect(
            plotArea.x - horzBleed / 2,
            // Allow bleed into padding at top.
            viewport.y,
            plotArea.width + horzBleed,
            plotArea.y + plotArea.height,
          )
          return path
        }),
      )

      const drawParts = () => {
        const drawPart = (
          { fillPattern }: (typeof style.parts)[0],
          i: number,
        ) => {
          const pixelXYs = pixelBands.map(({ x, ys }) => ({
            x,
            yRange: {
              start: Math.max(ys[i], i === 0 ? y0 : ys[i - 1]),
              end: ys[i],
            },
          }))
          ChartStyling.StripePattern.apply(
            ctx,
            fillPattern,
            null,
            ChartUtils.getRangePath(pixelXYs),
            fGet(this._stripePatternPathFn)(fillPattern),
          )
        }
        const n = style.parts.length
        // Draw the last part (the "rest") first because it can cut into
        // the other parts when it is below them. It's a subtle visual artifact,
        // but noticeable.
        drawPart(style.parts[n - 1], n - 1)
        style.parts.slice(0, n - 1).forEach(drawPart)
      }

      // Draw the parts below total.
      ChartUtils.saveRestore(ctx, () => {
        ctx.clip(
          ChartUtils.getRangePath(
            pixelBands.map(({ x, ys }) => ({
              x,
              yRange: { start: y0, end: ys[ys.length - 1] },
            })),
          ),
        )
        drawParts()
      })

      // Draw the parts above total.
      ChartUtils.saveRestore(ctx, () => {
        ctx.clip(
          ChartUtils.getRangePath(
            pixelBands.map(({ x, ys }) => ({
              x,
              yRange: { start: ys[ys.length - 1], end: viewport.y },
            })),
          ),
        )
        ctx.globalAlpha = 0.3
        drawParts()
      })

      // Draw the total line.
      ChartStyling.Stroke.apply(
        ctx,
        style.total.stroke,
        block(() => {
          const path = new Path2D()
          path.moveTo(
            pixelBands[0].x,
            pixelBands[0].ys[pixelBands[0].ys.length - 1],
          )
          pixelBands.forEach(({ x, ys }) => {
            path.lineTo(x, ys[ys.length - 1])
          })
          return path
        }),
      )
    })

    // Draw the pointer.
    ChartUtils.saveRestore(ctx, () => {
      if (currState.pointer.hover === 0) return
      const currYRanges = pointerTargetYRanges(
        currState.pointer.position.dataNotRounded.x,
      ).map(({ pixelYTransition, id }) => ({
        id,
        start: interpolate(pixelYTransition.start),
        end: interpolate(pixelYTransition.end),
      }))
      if (currYRanges.length === 0) return
      ctx.globalAlpha = currState.pointer.hover

      // Pointer.
      // const gap = 3/2 + .5
      const gap = 0.75
      currYRanges.forEach((yRange, i) => {
        if (yRange.end >= yRange.start) return
        const adjustStrokeWidth = (stroke: ChartStyling.Stroke) => ({
          ...stroke,
          width: stroke.width * currState.pointer.hover,
        })

        ChartStyling.Stroke.apply(
          ctx,
          adjustStrokeWidth(
            interpolate({
              from: style.parts[i].pointer.notHover,
              target: style.parts[i].pointer.hover,
              progress: letIn(
                this._partHoverAnimationMap.get(yRange.id),
                (animation) =>
                  animation ? interpolate(animation.transition) : 0,
              ),
            }),
          ),
          block(() => {
            const path = new Path2D()
            path.moveTo(currState.pointer.position.visual.x, yRange.start - gap)
            path.lineTo(currState.pointer.position.visual.x, yRange.end + gap)
            return path
          }),
        )
      })
    })

    return (dataX: number) => {
      const yRanges = pointerTargetYRanges(dataX)
      if (yRanges.length === 0) return []
      return [
        ...yRanges.map(({ id, pixelYTransition }) => ({
          id,
          pixelYTransition: pixelYTransition.mid,
          hoverTransition: letIn(
            this._partHoverAnimationMap.get(id),
            (animation) => (animation ? animation.transition : null),
          ),
        })),
        {
          id: this.ids.total,
          pixelYTransition: fGet(_.last(yRanges)).pixelYTransition.end,
          hoverTransition: null,
        },
      ]
    }
  }

  public get ids() {
    return {
      component: this._id,
      part: (partId: string) => `${this._id}-part-${partId}`,
      remaining: `${this._id}-remaining`,
      total: `${this._id}-total`,
    }
  }
}

const _getYRangeChecked = (
  dataX: number,
  data: {
    yByX: (number | null)[] | Float64Array
    xRange: SimpleRange | null
  },
) => (SimpleRange.Closed.isIn(dataX, data.xRange) ? fGet(data.yByX[dataX]) : 0)
