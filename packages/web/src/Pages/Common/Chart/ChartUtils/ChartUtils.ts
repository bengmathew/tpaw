import { fGet } from '../../../../Utils/Utils'

export type Rect = {
  x: number
  y: number
  width: number
  height: number
}
export type RectExt = Rect & {right: number; bottom: number}

export const rectExt = (
  rect:
    | {
        x: number
        y: number
        width: number
        height: number
      }
    | {
        x: number
        y: number
        width: number
        bottom: number
      }
): RectExt => {
  if ('bottom' in rect) {
    return {
      ...rect,
      height: rect.bottom - rect.y,
      right: rect.x + rect.width,
    }
  } else {
    return {
      ...rect,
      right: rect.x + rect.width,
      bottom: rect.y + rect.height,
    }
  }
}

export type Size = {width: number; height: number}
export namespace ChartUtils {
  export const getFont = (
    size: number,
    style: 'bold' | '800' | '900' | '' = ''
  ) => `${style} ${size}px Montserrat, sans-serif`
  export const getMonoFont = (
    size: number,
    style: 'bold' | '800' | '900' | '' = ''
  ) => `${style} ${size}px Roboto Mono, monospace`
  export const color = {
    gray: {
      '0': '#FFFFFF',
      '50': '#F9FAFB',
      '100': '#F3F4F6',
      '200': '#E5E7EB',
      '300': '#D1D5DB',
      '400': '#9CA3AF',
      '500': '#6B7280',
      '600': '#4B5563',
      '700': '#374151',
      '800': '#1F2937',
      '900': '#111827',
      '1000': '#000000',
    },
    yellow: {
      '500': '#F59E0B',
    },
    indigo: {
      '50': '#EFF6FF',
      '100': '#DBEAFE',
      '200': '#BFDBFE',
      '300': '#93C5FD',
      '400': '#60A5FA',
      '500': '#3B82F6',
      '600': '#2563EB',
      '700': '#1D4ED8',
      '800': '#1E40AF',
      '900': '#1E3A8A',
    },
    red: {
      '50': '#FEF2F2',
      '100': '#FEE2E2',
      '200': '#FECACA',
      '300': '#FCA5A5',
      '400': '#F87171',
      '500': '#EF4444',
      '600': '#DC2626',
      '700': '#B91C1C',
      '800': '#991B1B',
      '900': '#7F1D1D',
    },
    orange: {
      '50': '#fff7ed',
      '100': '#ffedd5',
      '200': '#fed7aa',
      '300': '#fdba74',
      '400': '#fb923c',
      '500': '#f97316',
      '600': '#ea580c',
      '700': '#c2410c',
      '800': '#9a3412',
      '900': '#7c2d12',
    },
    teal: {
      '50': '#f0fdfa',
      '100': '#ccfbf1',
      '200': '#99f6e4',
      '300': '#5eead4',
      '400': '#2dd4bf',
      '500': '#14b8a6',
      '600': '#0d9488',
      '700': '#0f766e',
      '800': '#115e59',
      '900': '#134e4a',
    },
    theme1Light: '#14b8a6', // teal 500
    theme1: '#0d9488', // teal 600
    theme1Dark: '#0f766e', // teal 700
    withOpacity(color: string, v: number) {
      return `${color}${Math.round(v * 255)
        .toString(16)
        .padStart(2, '0')}`
    },
  }

  export function roundRect(
    ctx: CanvasRenderingContext2D,
    {x, y, width, height}: Rect,
    radius: number
  ) {
    const right = x + width
    const bottom = y + height
    const edges = [
      {x, y},
      {x: right, y},
      {x: right, y: bottom},
      {x, y: bottom},
    ]
    roundedPolygon(ctx, edges, radius)
  }

  // Thanks, https://stackoverflow.com/a/56214413
  type _Point = {x: number; y: number}
  export function roundedPolygon(
    ctx: CanvasRenderingContext2D,
    points: _Point[],
    radius: number
  ) {
    const corners = _roundedPolygonCorners(points, radius)
    ctx.moveTo(corners[0].start.x, corners[0].start.y)
    for (const {start, ctrl, end} of corners) {
      ctx.lineTo(start.x, start.y)
      ctx.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    }
    ctx.closePath()
  }

  export function roundedLine(
    ctx: CanvasRenderingContext2D,
    points: _Point[],
    radius: number
  ) {
    // const [last, ...corners] = _roundedPolygonCorners(points, radius)
    // ctx.moveTo(last[1].x, last[1].y)
    // for (const [start, ctrl, end] of corners.slice(0, -1)) {
    //   ctx.lineTo(start.x, start.y)
    //   ctx.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    // }
    // ctx.lineTo(points[0].x, points[0].y)

    const corners = _roundedPolygonCorners(points, radius)
    const first = fGet(corners.pop())
    const last = fGet(corners.pop())
    ctx.moveTo(first.ctrl.x, first.ctrl.y)
    for (const {start, ctrl, end} of corners) {
      ctx.lineTo(start.x, start.y)
      ctx.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    }
    ctx.lineTo(last.ctrl.x, last.ctrl.y)
  }

  function _roundedPolygonCorners(points: _Point[], radius: number) {
    const distance = (p1: _Point, p2: _Point) =>
      Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    const lerp = (a: number, b: number, x: number) => a + (b - a) * x

    const lerp2D = (p1: _Point, p2: _Point, t: number) => ({
      x: lerp(p1.x, p2.x, t),
      y: lerp(p1.y, p2.y, t),
    })

    const numPoints = points.length

    const corners = []
    for (let i = 0; i < numPoints; i++) {
      const lastPoint = points[i]
      const thisPoint = points[(i + 1) % numPoints]
      const nextPoint = points[(i + 2) % numPoints]

      const lastEdgeLength = distance(lastPoint, thisPoint)
      const lastOffsetDistance = Math.min(lastEdgeLength / 2, radius)
      const start = lerp2D(
        thisPoint,
        lastPoint,
        lastOffsetDistance / lastEdgeLength
      )

      const nextEdgeLength = distance(nextPoint, thisPoint)
      const nextOffsetDistance = Math.min(nextEdgeLength / 2, radius)
      const end = lerp2D(
        thisPoint,
        nextPoint,
        nextOffsetDistance / nextEdgeLength
      )

      corners.push({start, ctrl: thisPoint, end})
    }
    return corners
  }
}
