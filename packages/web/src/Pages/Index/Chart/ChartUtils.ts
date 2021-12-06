export type Rect = {x: number; y: number; width: number; height: number}
export type Size = {width: number; height: number}
export namespace ChartUtils {
  export const getFont = (size: number) => `${size}px Montserrat, sans-serif`
  export const color = {
    gray: {
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
  }

  export function roundRect(
    ctx: CanvasRenderingContext2D,
    {x, y, width, height}: Rect,
    radius: number
  ) {
    ctx.moveTo(x + radius, y)
    ctx.lineTo(x + width - radius, y)
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius)
    ctx.lineTo(x + width, y + height - radius)
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height)
    ctx.lineTo(x + radius, y + height)
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius)
    ctx.lineTo(x, y + radius)
    ctx.quadraticCurveTo(x, y, x + radius, y)
    ctx.closePath()
  }
}
