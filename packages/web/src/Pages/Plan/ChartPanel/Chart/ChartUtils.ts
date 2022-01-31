export type Rect = {x: number; y: number; width: number; height: number}
export type Size = {width: number; height: number}
export namespace ChartUtils {
  export const getFont = (size: number, style: 'bold' | '' = '') =>
    `${style} ${size}px Montserrat, sans-serif`
  export const getMonoFont = (size: number, style: 'bold' | '' = '') =>
    `${style} ${size}px Roboto Mono, monospace`
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
