import _ from 'lodash'
import { RGB, RGBA } from '../../../../Utils/ColorUtils'
import { Rect, XY } from '../../../../Utils/Geometry'
import { ChartUtils } from './ChartUtils'

export namespace ChartStyling {
  // --------------
  // ---- Stroke ----
  // --------------
  export type Stroke = { color: RGB; width: number; opacity: number }
  export namespace Stroke {
    export const get = (color: RGB, width: number, opacity = 1): Stroke => ({
      color,
      width,
      opacity,
    })
    export const deNull = (
      src: Stroke | null,
      other: Stroke | null,
      opts?: { fixedWidth?: boolean },
    ): Stroke => {
      const fixedWidth = opts?.fixedWidth ?? false
      return (
        src ?? {
          color: other?.color ?? [0, 0, 0],
          width: fixedWidth ? other?.width ?? 0 : 0,
          opacity: 0,
        }
      )
    }

    export const apply = (
      ctx: CanvasRenderingContext2D,
      { color, width, opacity }: Stroke,
      path: Path2D,
    ) => {
      if (width === 0 || opacity === 0) return
      ChartUtils.saveRestore(ctx, () => {
        ctx.lineWidth = width
        ctx.strokeStyle = RGB.toHex(color)
        ctx.globalAlpha *= opacity ?? 1
        ctx.stroke(path)
      })
    }
  }

  // ----------------
  // ---- Shadow ----
  // ----------------
  export type Shadow = { color: RGBA; blur: number; offset: XY }
  export namespace Shadow {
    export const withDefaultColor = (x: Omit<Shadow, 'color'>): Shadow => ({
      ...x,
      color: [0, 0, 0, 0.2],
    })
    export const deNull = (src: Shadow | null, other: Shadow | null): Shadow =>
      src ?? {
        color: other?.color ?? [0, 0, 0, 0],
        blur: 0,
        offset: other?.offset ?? { x: 0, y: 0 },
      }
    export const fill = (
      ctx: CanvasRenderingContext2D,
      shadow: Shadow | null,
      fill: { color: RGB; opacity?: number },
      path: Path2D,
    ) => {
      ChartUtils.saveRestore(ctx, () => {
        ctx.globalAlpha *= fill.opacity ?? 1
        ctx.fillStyle = RGB.toHex(fill.color)
        if (shadow && shadow.blur !== 0) {
          ctx.shadowColor = RGB.toHex(shadow.color)
          ctx.shadowBlur = shadow.blur
          ctx.shadowOffsetX = shadow.offset.x
          ctx.shadowOffsetY = shadow.offset.y
        }
        ctx.fill(path)
      })
    }
  }

  // ------------------------
  // ---- Stripe Pattern ----
  // ------------------------
  export type StripePattern = {
    bg: { color: RGB; opacity: number }
    stroke: Stroke
    gap: number
  }

  export namespace StripePattern {
    export type PatternPathFn = ReturnType<typeof getPatternPathFn>
    export const getPatternPathFn = (viewport: Rect) => {
      type Args = {
        strokeWidth: number
        gap: number
      }
      const memoized = _.memoize((argsStr: string) => {
        const { strokeWidth, gap } = JSON.parse(argsStr) as Args
        const strokePath = new Path2D()

        const width = strokeWidth + gap
        _.range(viewport.x, viewport.x + viewport.width, width).forEach(
          (pixelX) => {
            strokePath.moveTo(pixelX, viewport.y)
            strokePath.lineTo(pixelX, viewport.y + viewport.height)
          },
        )
        return strokePath
      })
      const result = ({ stroke, gap }: StripePattern) => {
        const args: Args = { strokeWidth: stroke.width, gap }
        return memoized(JSON.stringify(args))
      }
      result.destroy = () => memoized.cache.clear?.()
      return result
    }
    export const apply = (
      ctx: CanvasRenderingContext2D,
      { bg, stroke, gap }: StripePattern,
      shadow: Shadow | null,
      path: Path2D,
      patternPath: Path2D,
    ) =>
      ChartUtils.saveRestore(ctx, () => {
        Shadow.fill(ctx, shadow, bg, path)
        ctx.clip(path)
        Stroke.apply(ctx, stroke, patternPath)
      })
  }
}
