import { block } from '@tpaw/common'
import { ChartStyling } from '../Pages/Common/Chart/ChartUtils/ChartStyling'
import { RGB, tint } from './ColorUtils'

export const gray = {
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
}
export const slate = {
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
}
export const yellow = {
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
}
export const indigo = {
  '50': '#eef2ff',
  '100': '#e0e7ff',
  '200': '#c7d2fe',
  '300': '#a5b4fc',
  '400': '#818cf8',
  '500': '#6366f1',
  '600': '#4f46e5',
  '700': '#4338ca',
  '800': '#3730a3',
  '900': '#312e81',
  '950': '#1e1b4b',
}
export const purple = {
  '50': '#faf5ff',
  '100': '#f3e8ff',
  '200': '#e9d5ff',
  '300': '#d8b4fe',
  '400': '#c084fc',
  '500': '#a855f7',
  '600': '#9333ea',
  '700': '#7e22ce',
  '800': '#6b21a8',
  '900': '#581c87',
}
export const cyan = {
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
}
export const sky = {
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
}
export const blue = {
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
}
export const violet = {
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
}
export const red = {
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
}
export const rose = {
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
}
export const pink = {
  '50': '#fdf2f8',
  '100': '#fce7f3',
  '200': '#fbcfe8',
  '300': '#f9a8d4',
  '400': '#f472b6',
  '500': '#ec4899',
  '600': '#db2777',
  '700': '#be185d',
  '800': '#9d174d',
  '900': '#831843',
}
export const orange = {
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
}
export const teal = {
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
}
export const emerald = {
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
  fuchsia = {
    '50': '#fdf4ff',
    '100': '#fae8ff',
    '200': '#f5d0fe',
    '300': '#f0abfc',
    '400': '#e879f9',
    '500': '#d946ef',
    '600': '#c026d3',
    '700': '#a21caf',
    '800': '#86198f',
    '900': '#701a75',
    '950': '#4a044e',
  }
export const green = {
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
}
export const amber = {
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
}
export const lime = {
  50: '#f7fee7',
  100: '#ecfccb',
  200: '#d9f99d',
  300: '#bef264',
  400: '#a3e635',
  500: '#84cc16',
  600: '#65a30d',
  700: '#4d7c0f',
  800: '#3f6212',
  900: '#365314',
  950: '#1a2e05',
}

export const colorPalette = {
  withOpacity(color: string, v: number) {
    return `${color}${Math.round(v * 255)
      .toString(16)
      .padStart(2, '0')}`
  },
  gray,
  slate,
  yellow,
  indigo,
  purple,
  cyan,
  sky,
  blue,
  violet,
  red,
  rose,
  pink,
  orange,
  teal,
  emerald,
  fuchsia,
  green,
  amber,
  lime,
}

const { Stroke } = ChartStyling

const _chartBandColors = block(() => {
  const colorNames = block(() => {
    const ordered = [
      'violet',
      'pink',
      'cyan',
      'orange',
      'green',
      'red',
      'fuchsia',
      'indigo',
      'yellow',
      'teal',
      'sky',
      'purple',
      'lime',
      'blue',
      'rose', // very much like red
      'amber', // very much like orange
      'emerald', // very much like green
    ] as const
    return ordered
  })

  return colorNames.map((color: (typeof colorNames)[number]) => {
    return {
      fillPattern: {
        bg: {
          color: tint(0.1, RGB.fromHex(colorPalette[color][400])),
          opacity: 1,
        },
        stroke: Stroke.get(RGB.fromHex(colorPalette[color][600]), 1),
        gap: 3,
      },
      pointer: {
        hover: Stroke.get(RGB.fromHex(colorPalette[color][700]), 10),
        notHover: Stroke.get(RGB.fromHex(colorPalette[color][700]), 3),
      },
    }
  })
})

export const getChartBandColor = (index: number) =>
  _chartBandColors[index % _chartBandColors.length]
