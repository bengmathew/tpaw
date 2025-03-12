import { block, noCase } from '@tpaw/common'
import { gray } from '../../../../../../Utils/ColorPalette'
import { RGB } from '../../../../../../Utils/ColorUtils'
import { Record } from '@tpaw/common'
import { ChartRange } from '../../../../../Common/Chart/ChartComponent/ChartRange'
import { ChartStyling } from '../../../../../Common/Chart/ChartUtils/ChartStyling'
import { PlanResultsChartDataForPDF } from './PlanResultsChartData'

const { Stroke } = ChartStyling

export const getPlanResultsChartRange = (
  mode: 'main' | 'menu' | 'inline' | 'print',
) =>
  new ChartRange<{ data: PlanResultsChartDataForPDF }>('range', ({ data }) => {
    const { planColors } = data
    const { shades } = planColors
    const byMode = block(() => {
      switch (mode) {
        case 'main':
          return {
            pattern: { strokeW: 1, gap: 3 },
            midLine: { w: 2.5, color: shades.main[13].rgb },
          }
        case 'print':
          return {
            pattern: { strokeW: 1, gap: 3 },
            midLine: { w: 2, color: shades.main[13].rgb },
          }
        case 'inline':
          return {
            pattern: { strokeW: 1, gap: 3 },
            midLine: { w: 1.5, color: RGB.fromHex(gray['500']) },
          }
        case 'menu':
          return {
            pattern: { strokeW: 1, gap: 1.5 },
            midLine: { w: 1, color: shades.main[13].rgb },
          }
        default:
          noCase(mode)
      }
    })
    const style = {
      pointer: {
        line: {
          inside: Stroke.get(shades.main[10].rgb, 3),
          outside: Stroke.get(shades.main[5].rgb, 1),
        },
        dot: {
          start: { color: shades.main[8].rgb, opacity: 1, radius: 5.5 },
          mid: { color: shades.main[8].rgb, opacity: 1, radius: 5.5 },
          end: { color: shades.main[8].rgb, opacity: 1, radius: 5.5 },
        },
      },
      fillPattern: {
        bg: { color: shades.light[5].rgb, opacity: 1 },
        stroke: Stroke.get(shades.main[4].rgb, byMode.pattern.strokeW),
        gap: byMode.pattern.gap,
      },
      lines: {
        start: Stroke.get(shades.main[3].rgb, 0, 0),
        mid: Stroke.get(byMode.midLine.color, byMode.midLine.w, 1),
        end: Stroke.get(shades.main[3].rgb, 0, 0),
      },
    }
    return data.type === 'range'
      ? { range: data.range, style }
      : {
          range: data.breakdown.range,
          style: {
            pointer: {
              line: {
                inside: { ...style.pointer.line.inside, opacity: 0 },
                outside: { ...style.pointer.line.outside, opacity: 0 },
              },
              dot: Record.mapValues(style.pointer.dot, (v) => ({
                ...v,
                opacity: 0,
                radius: 3,
              })),
            },
            fillPattern: {
              bg: { ...style.fillPattern.bg, opacity: 0.15 },
              stroke: { ...style.fillPattern.stroke, opacity: 0.15 },
              gap: style.fillPattern.gap,
            },
            lines: {
              start: Stroke.get(shades.main[3].rgb, 1.1),
              mid: Stroke.get(shades.main[3].rgb, 1.1),
              end: Stroke.get(shades.main[3].rgb, 1.1),
            },
          },
        }
  })
