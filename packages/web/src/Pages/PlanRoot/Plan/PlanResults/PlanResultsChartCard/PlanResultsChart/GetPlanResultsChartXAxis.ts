import { fGet } from '@tpaw/common'
import { ChartXAxis } from '../../../../../Common/Chart/ChartComponent/ChartXAxis'
import { PlanResultsChartDataForPDF } from './PlanResultsChartData'

export const getPlanResultsChartXAxis = (
  personType: 'person1' | 'person2',
  belowPlotAreaSizing: {
    gapToLine: number
    lineWidth: number
    xAxis: { height: number; vertGapBetweenPeople: number }
  },
) =>
  new ChartXAxis<{ data: PlanResultsChartDataForPDF }>(
    ({ data: { planParamsNorm, layout, planColors } }) => {
      const { ages } = planParamsNorm
      if (personType === 'person2' && !ages.person2) return null
      const person = fGet(ages[personType])
      return {
        colors: planColors,
        marginTop:
          belowPlotAreaSizing.gapToLine +
          belowPlotAreaSizing.lineWidth +
          (personType === 'person2'
            ? belowPlotAreaSizing.xAxis.height +
              belowPlotAreaSizing.xAxis.vertGapBetweenPeople
            : 0),
        padding: { top: personType === 'person1' ? 0 : 1 },
        height: belowPlotAreaSizing.xAxis.height,
        transformDataXDelta: person.currentAgeInfo.inMonths,
        markers: {
          start: 0,
          retirement: person.retirement.ageIfInFuture?.asMFN ?? 0,
          end: person.maxAge.asMFN,
        },
        label: {
          text: personType === 'person1' ? 'Your Age' : `Partner's Age`,
          fontSize: layout === 'mobile' ? 12 : 14,
          paddingLeft: layout === 'mobile' ? 15 : 20,
        },
        pointer: {
          height: layout === 'mobile' ? 6 : 7,
        },
        getTick: (ageInMonths, pixelsPerMonth) => {
          if (ageInMonths % 12 !== 0) return null
          const pixelsPerYear = pixelsPerMonth * 12
          const text = `${ageInMonths / 12}`
          const ageInYears = Math.round(ageInMonths / 12)
          return ageInYears % 10 === 0
            ? {
                color: planColors.shades.main[12].hex,
                line: {
                  length: layout === 'mobile' ? 5 : 6,
                },
                label: {
                  text,
                  paddingTop: 2,
                  fontSize: layout === 'mobile' ? 9 : 10,
                  fontWeight: '500',
                  visible: true,
                },
              }
            : ageInYears % 5 === 0
              ? {
                  color: planColors.shades.main[13].hex,
                  line: {
                    length: layout === 'mobile' ? 3 : 5,
                  },
                  label: {
                    text,
                    paddingTop: 2,
                    fontSize: layout === 'mobile' ? 8 : 9,
                    fontWeight: '',
                    visible: pixelsPerYear > 15,
                  },
                }
              : {
                  color: planColors.shades.main[14].hex,
                  line: {
                    length: layout === 'mobile' ? 1 : 2,
                  },
                  label: {
                    text,
                    paddingTop: 2,
                    fontSize: layout === 'mobile' ? 8 : 9,
                    fontWeight: '',
                    visible: false,
                  },
                }
        },
      }
    },
  )
