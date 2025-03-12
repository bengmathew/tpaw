import { block, fGet } from '@tpaw/common'
import { RGB } from '../../../../Utils/ColorUtils'
import { interpolate } from '../../../../Utils/Interpolate'
import { Record } from '@tpaw/common'
import { SimpleRange } from'@tpaw/common'
import { transitionTransform } from '../../../../Utils/Transition'
import { ChartContext } from '../ChartContext'
import { ChartStyling } from '../ChartUtils/ChartStyling'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export type ChartRangeProps = {
  range: {
    yRangeByX: (null | (SimpleRange & { mid: number }))[]
    xRange: SimpleRange | null // Null means not valid anywhere.
  }
  style: {
    pointer: {
      line: Record<'inside' | 'outside', ChartStyling.Stroke | null> | null
      dot: {
        start: { color: RGB; opacity: number; radius: number } | null
        mid: { color: RGB; opacity: number; radius: number } | null
        end: { color: RGB; opacity: number; radius: number } | null
      } | null
    } | null
    fillPattern: ChartStyling.StripePattern
    lines: Record<'start' | 'mid' | 'end', ChartStyling.Stroke | null> | null
  }
}

const _deNullProps = (src: ChartRangeProps, other: ChartRangeProps) => ({
  getYRange: (dataX: number) =>
    SimpleRange.Closed.isIn(dataX, src.range.xRange)
      ? fGet(src.range.yRangeByX[dataX])
      : { start: 0, mid: 0, end: 0 },
  style: {
    pointer: {
      line: {
        inside: ChartStyling.Stroke.deNull(
          src.style.pointer?.line?.inside ?? null,
          other.style.pointer?.line?.inside ?? null,
        ),
        outside: ChartStyling.Stroke.deNull(
          src.style.pointer?.line?.outside ?? null,
          other.style.pointer?.line?.outside ?? null,
        ),
      },
      dot: (['start', 'mid', 'end'] as const).map((l) => ({
        color: src.style.pointer?.dot?.[l]?.color ??
          other.style.pointer?.dot?.[l]?.color ?? [0, 0, 0],
        opacity: src.style.pointer?.dot?.[l]?.opacity ?? 0,
        radius: src.style.pointer?.dot?.[l]?.radius ?? 0,
      })),
      // mid: {
      //   color: src.style.pointer?.dot?.mid?.color ??
      //     other.style.pointer?.dot?.mid?.color ?? [0, 0, 0],
      //   opacity: src.style.pointer?.dot?.mid?.opacity ?? 0,
      //   radius: src.style.pointer?.dot?.mid?.radius ?? 0,
      // },
      // end: {
      //   color: src.style.pointer?.dot?.end?.color ??
      //     other.style.pointer?.dot?.end?.color ?? [0, 0, 0],
      //   opacity: src.style.pointer?.dot?.end?.opacity ?? 0,
      //   radius: src.style.pointer?.dot?.end?.radius ?? 0,
      // },
      // },
    },
    fillPattern: src.style.fillPattern,
    lines: Record.mapValues(
      src.style.lines ?? { start: null, mid: null, end: null },
      (x, key) =>
        ChartStyling.Stroke.deNull(x ?? null, other.style.lines?.[key] ?? null),
    ),
  },
})

export class ChartRange<Params> implements ChartComponent<Params> {
  private _stripePatternPathFn: ChartStyling.StripePattern.PatternPathFn | null =
    null
  constructor(
    private _id: string,
    private _propsFn: (params: Params) => ChartRangeProps,
  ) {}

  get ids() {
    return {
      start: `${this._id}-start`,
      mid: `${this._id}-mid`,
      end: `${this._id}-end`,
    }
  }
  draw(
    context: ChartContext<Params>,
    change:
      | 'draw'
      | 'init'
      | 'pointer'
      | 'stateAndPointer'
      | 'sizingAndPointer',
  ) {
    const { canvasContext: ctx, stateTransition, currState } = context
    const { derivedState } = currState
    const { plotArea, viewport } = derivedState
    if (change !== 'draw') {
      this._stripePatternPathFn?.destroy()
      this._stripePatternPathFn = ChartStyling.StripePattern.getPatternPathFn(
        context.stateTransition.target.derivedState.viewport,
      )
    }

    const directTransition = transitionTransform(stateTransition, (x) => ({
      ...x,
      props: this._propsFn(x.params),
    }))

    const fullTransition = transitionTransform(
      directTransition,
      (src, other) => ({
        ...src,
        props: _deNullProps(src.props, other.props),
      }),
    )

    const dataXRange = SimpleRange.union(
      directTransition.from.props.range.xRange,
      directTransition.target.props.range.xRange,
    )
    if (dataXRange === null) return null
    const dataXs = SimpleRange.Closed.toArr(dataXRange)

    const pixelTransition = transitionTransform(
      fullTransition,
      ({ derivedState: { scale }, props }, __, src) => ({
        pixelXYs: dataXs.map((dataX: number) => ({
          x: scale.x.start(dataX),
          yRange: Record.mapValues(props.getYRange(dataX), scale.y),
        })),
        style: props.style,
      }),
    )

    const { pixelXYs, style } = interpolate(pixelTransition)
    const pointerTargetY = (dataX: number) => {
      const dataXOffset = dataX - dataXRange.start
      if (dataXOffset < 0 || dataXOffset > pixelXYs.length - 1) return []
      const floor = Math.floor(dataXOffset)
      const ceil = Math.ceil(dataXOffset)

      return (['start', 'mid', 'end'] as const).map((part) => ({
        id: this.ids[part],
        pixelYTransition: transitionTransform(
          pixelTransition,
          ({ pixelXYs }) =>
            floor === ceil
              ? pixelXYs[floor].yRange[part]
              : interpolate({
                  from: pixelXYs[floor].yRange[part],
                  target: pixelXYs[ceil].yRange[part],
                  progress: (dataXOffset - floor) / (ceil - floor),
                }),
        ),
        hoverTransition: null,
      }))
    }

    ChartUtils.saveRestore(ctx, () => {
      // Apply the clip.
      ctx.clip(
        block(() => {
          const path = new Path2D()
          const lineWidth = Math.max(
            style.lines.start.width,
            style.lines.end.width,
            style.lines.mid.width,
          )
          path.rect(
            plotArea.x - lineWidth / 2,
            // Allow bleed into padding at top.
            viewport.y,
            plotArea.width + lineWidth,
            plotArea.y + plotArea.height + lineWidth / 2,
          )
          return path
        }),
      )

      // Draw the fill.
      ChartUtils.saveRestore(ctx, () => {
        ChartStyling.StripePattern.apply(
          ctx,
          style.fillPattern,
          null,
          ChartUtils.getRangePath(pixelXYs),
          fGet(this._stripePatternPathFn)(style.fillPattern),
        )
      })

      // Draw the lines.
      ;(['start', 'mid', 'end'] as const).map((part) => {
        const lineStyle = style.lines[part]
        if (lineStyle.width === 0 || lineStyle.opacity === 0) return
        ChartUtils.saveRestore(ctx, () => {
          const path = new Path2D()
          path.moveTo(pixelXYs[0].x, pixelXYs[0].yRange[part])
          pixelXYs.forEach((pixelXY) => {
            path.lineTo(pixelXY.x, pixelXY.yRange[part])
          })
          ChartStyling.Stroke.apply(ctx, lineStyle, path)
        })
      })
    })

    // Draw the pointer
    ChartUtils.saveRestore(ctx, () => {
      if (currState.pointer.hover === 0) return
      const pointerPixelX = currState.pointer.position.visual.x
      const ys = pointerTargetY(
        currState.pointer.position.dataNotRounded.x,
      ).map(({ pixelYTransition }) => interpolate(pixelYTransition))
      if (ys.length === 0) return
      ChartUtils.saveRestore(ctx, () => {
        ctx.globalAlpha = currState.pointer.hover
        const adjustStrokeWidth = (stroke: ChartStyling.Stroke) => ({
          ...stroke,
          width: stroke.width * currState.pointer.hover,
        })
        {
          const path = new Path2D()
          path.moveTo(pointerPixelX, ys[0])
          path.lineTo(pointerPixelX, plotArea.y + plotArea.height)
          ChartStyling.Stroke.apply(
            ctx,
            adjustStrokeWidth(style.pointer.line.outside),
            path,
          )
        }
        {
          const path = new Path2D()
          path.moveTo(pointerPixelX, ys[0])
          path.lineTo(pointerPixelX, ys[2])
          ChartStyling.Stroke.apply(
            ctx,
            adjustStrokeWidth(style.pointer.line.inside),
            path,
          )
        }

        // Draw the dots.
        ys.map((y, i) => {
          ChartUtils.saveRestore(ctx, () => {
            const dotStyle = style.pointer.dot[i]
            const r = dotStyle.radius * currState.pointer.hover
            const path = new Path2D()
            path.ellipse(pointerPixelX, y, r, r, 0, 0, Math.PI * 4)
            ctx.globalAlpha *= dotStyle.opacity
            ctx.fillStyle = RGB.toHex(dotStyle.color)
            ctx.fill(path)
          })
        })
      })
    })

    return pointerTargetY
  }
}
