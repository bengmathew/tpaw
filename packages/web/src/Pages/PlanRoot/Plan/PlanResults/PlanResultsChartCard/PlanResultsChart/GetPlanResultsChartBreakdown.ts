import { getChartBandColor } from '../../../../../../Utils/ColorPalette'
import { ChartBreakdown } from '../../../../../Common/Chart/ChartComponent/ChartBreakdown'
import { ChartStyling } from '../../../../../Common/Chart/ChartUtils/ChartStyling'
import { PlanColors } from '../../../UsePlanColors'
import { PlanResultsChartData } from './PlanResultsChartData'

const { Stroke } = ChartStyling

export const getPlanResultsChartBreakdown = () =>
  new ChartBreakdown<{
    data: PlanResultsChartData
  }>('breakdown', ({ data }) => {
    const { planColors } = data
    const { shades } = planColors
    return data.type === 'breakdown'
      ? {
          total: {
            yByX: data.breakdown.total.yByX,
            xRange: data.breakdown.total.xRange,
          },
          parts: data.breakdown.parts.map((x) => ({
            id: x.id,
            data: x.data,
            style: {
              ...getChartBandColor(x.chartColorIndex),
            },
          })),
          style: {
            total: {
              fillPattern: getChartBreakdownTotalFillPattern(planColors),
              stroke: Stroke.get(shades.main[10].rgb, 2.5),
              pointer: {
                hover: Stroke.get(shades.main[10].rgb, 10),
                notHover: Stroke.get(shades.main[10].rgb, 3),
              },
            },
          },
        }
      : null
  })

export const getChartBreakdownTotalFillPattern = ({ shades }: PlanColors) => ({
  bg: { color: shades.light[5].rgb, opacity: 1 },
  stroke: Stroke.get(shades.main[5].rgb, 1),
  gap: 3,
})
