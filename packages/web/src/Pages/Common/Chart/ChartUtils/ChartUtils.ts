import {fGet} from '../../../../Utils/Utils'

export type Rect = {
  x: number
  y: number
  width: number
  height: number
}
export type RectExt = Rect & {right: number; bottom: number}

export const rectExt = (
  rect:
    | {x: number; y: number; width: number; height: number}
    | {x: number; y: number; width: number; bottom: number}
    | {x: number; y: number; right: number; height: number}
    | {x: number; y: number; right: number; bottom: number}
): RectExt => {
  const width = 'width' in rect ? rect.width : rect.right - rect.x
  const height = 'height' in rect ? rect.height : rect.bottom - rect.y
  const right = 'right' in rect ? rect.right : rect.width + rect.x
  const bottom = 'bottom' in rect ? rect.bottom : rect.height + rect.y
  const {x, y} = rect
  return {x, y, width, height, bottom, right}
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
    slate: {
      '50': '#f8fafc',
      '100': '#f1f5f9',
      '200': '#e2e8f0',
      '300': '#cbd5e1',
      '400': '#94a3b8',
      '500': '#64748b',
      '600': '#475569',
      '700': '#334155',
      '800': '#1e293b',
      '900': '#0f172a',
    },
    yellow: {
      '50': '#fefce8',
      '100': '#fef9c3',
      '200': '#fef08a',
      '300': '#fde047',
      '400': '#facc15',
      '500': '#eab308',
      '600': '#ca8a04',
      '700': '#a16207',
      '800': '#854d0e',
      '900': '#713f12',
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
    cyan: {
      '50': '#ecfeff',
      '100': '#cffafe',
      '200': '#a5f3fc',
      '300': '#67e8f9',
      '400': '#22d3ee',
      '500': '#06b6d4',
      '600': '#0891b2',
      '700': '#0e7490',
      '800': '#155e75',
      '900': '#164e63',
    },
    sky: {
      '50': '#f0f9ff',
      '100': '#e0f2fe',
      '200': '#bae6fd',
      '300': '#7dd3fc',
      '400': '#38bdf8',
      '500': '#0ea5e9',
      '600': '#0284c7',
      '700': '#0369a1',
      '800': '#075985',
      '900': '#0c4a6e',
    },
    blue: {
      '50': '#eff6ff',
      '100': '#dbeafe',
      '200': '#bfdbfe',
      '300': '#93c5fd',
      '400': '#60a5fa',
      '500': '#3b82f6',
      '600': '#2563eb',
      '700': '#1d4ed8',
      '800': '#1e40af',
      '900': '#1e3a8a',
    },
    violet: {
      '50': '#f5f3ff',
      '100': '#ede9fe',
      '200': '#ddd6fe',
      '300': '#c4b5fd',
      '400': '#a78bfa',
      '500': '#8b5cf6',
      '600': '#7c3aed',
      '700': '#6d28d9',
      '800': '#5b21b6',
      '900': '#4c1d95',
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
    rose: {
      '50': '#fff1f2',
      '100': '#ffe4e6',
      '200': '#fecdd3',
      '300': '#fda4af',
      '400': '#fb7185',
      '500': '#f43f5e',
      '600': '#e11d48',
      '700': '#be123c',
      '800': '#9f1239',
      '900': '#881337',
    },
    pink: {
      '∏': '#fdf2f8',
      '100': '#fce7f3',
      '200': '#fbcfe8',
      '300': '#f9a8d4',
      '400': '#f472b6',
      '500': '#ec4899',
      '600': '#db2777',
      '700': '#be185d',
      '800': '#9d174d',
      '900': '#831843',
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
    emerald: {
      '50': '#ecfdf5',
      '100': '#d1fae5',
      '200': '#a7f3d0',
      '300': '#6ee7b7',
      '400': '#34d399',
      '500': '#10b981',
      '600': '#059669',
      '700': '#047857',
      '800': '#065f46',
      '900': '#064e3b',
    },
    green: {
      '50': '#f0fdf4',
      '100': '#dcfce7',
      '200': '#bbf7d0',
      '300': '#86efac',
      '400': '#4ade80',
      '500': '#22c55e',
      '600': '#16a34a',
      '700': '#15803d',
      '800': '#166534',
      '900': '#14532d',
    },
    amber: {
      '50': '#fffbeb',
      '100': '#fef3c7',
      '200': '#fde68a',
      '300': '#fcd34d',
      '400': '#fbbf24',
      '500': '#f59e0b',
      '600': '#d97706',
      '700': '#b45309',
      '800': '#92400e',
      '900': '#78350f',
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
