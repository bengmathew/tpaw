import { block, fGet, letIn, noCase } from '@tpaw/common'
import _ from 'lodash'
import { getChartBandColor } from '../../../../../../Utils/ColorPalette'
import { RGB } from '../../../../../../Utils/ColorUtils'
import { Size, rectExt } from '../../../../../../Utils/Geometry'
import { interpolate } from '../../../../../../Utils/Interpolate'
import { SimpleRange } from '../../../../../../Utils/SimpleRange'
import { Transition } from '../../../../../../Utils/Transition'
import { ChartBreakdown } from '../../../../../Common/Chart/ChartComponent/ChartBreakdown'
import {
  ChartPointer,
  ChartPointerBoxComponents,
  ChartPointerProps,
} from '../../../../../Common/Chart/ChartComponent/ChartPointer'
import { ChartPointerPortal } from '../../../../../Common/Chart/ChartComponent/ChartPointerPortal'
import { ChartRange } from '../../../../../Common/Chart/ChartComponent/ChartRange'
import { ChartStyling } from '../../../../../Common/Chart/ChartUtils/ChartStyling'
import { ChartUtils } from '../../../../../Common/Chart/ChartUtils/ChartUtils'
import { PlanColors } from '../../../UsePlanColors'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
} from '../../PlanResultsChartType'
import { PlanResultsChartData } from './PlanResultsChartData'

const { Stroke } = ChartStyling

const _isInRange = SimpleRange.Closed.isIn

export const getPlanResultsChartPointer = (
  portal: ChartPointerPortal,
  rangeIds: ChartRange<unknown>['ids'],
  breakdownIds: ChartBreakdown<unknown>['ids'],
) =>
  new ChartPointer<{ data: PlanResultsChartData }>(
    portal,
    ({ params: { data }, derivedState }) => {
      const { planColors, planTransitionState, planSizing } = data
      const { shades } = planColors
      const { positioning, maxSize } = block(() => {
        const portalSizing =
          planTransitionState.section === 'help'
            ? planSizing.pointer.fixed.help
            : planTransitionState.section === 'summary'
              ? planSizing.pointer.fixed.summary
              : planSizing.pointer.fixed.input

        switch (planSizing.args.layout) {
          case 'laptop': {
            const gap = 10
            const positioning: ChartPointerProps['positioning'] = ({
              width,
              height,
              hover,
            }) => ({
              origin: {
                x: interpolate({
                  from: portalSizing.chartPosition.left + 1,
                  target: portalSizing.chartPosition.left - width - gap,
                  progress: hover,
                }),
                y: Math.max(
                  gap,
                  portalSizing.chartPosition.bottom -
                    height -
                    derivedState.padding.bottom,
                ),
              },
              clip: rectExt({
                x: 0,
                y: 0,
                right: portalSizing.chartPosition.left,
                bottom: portalSizing.region.bottom,
              }),
            })

            return {
              positioning,
              maxSize: {
                width: Math.min(500, portalSizing.chartPosition.left - gap * 2),
                height: portalSizing.region.height - gap * 2,
              },
            }
          }
          case 'desktop':
          case 'mobile': {
            const gap = 5
            const x =
              portalSizing.chartPosition.left + derivedState.padding.left
            const positioning: ChartPointerProps['positioning'] = ({
              width,
              height,
              hover,
            }) => {
              return {
                origin: {
                  x,
                  y: interpolate({
                    from: portalSizing.chartPosition.bottom - height - 1,
                    target: portalSizing.chartPosition.bottom + gap,
                    progress: hover,
                  }),
                },
                clip: rectExt({
                  x: 0,
                  y: portalSizing.chartPosition.bottom,
                  bottom: portalSizing.region.bottom,
                  right: portalSizing.region.right,
                }),
              }
            }

            return {
              positioning,
              maxSize: {
                width: Math.min(500, portalSizing.region.width - x * 2),
                height:
                  portalSizing.region.height -
                  portalSizing.chartPosition.bottom -
                  gap * 2,
              },
            }
          }
          default:
            noCase(planSizing.args.layout)
        }
      })
      return {
        positioning,
        getBox: (dataX, hoverTransitionMap, components) => {
          const { box, group, style, gap, hr } = components

          const headerSection = _getHeaderSection(dataX, components, data)
          const headerHeight = headerSection().height
          const gapBelowHeaderSection = 10
          const hrLineHeight = 2
          const boxOpts = {
            padding: { vert: 15, horz: 20 },
            borderRadius: 10,
            lineWidth: 0,
          }
          return style(
            { fillColor: planColors.dark.hex },
            box(
              boxOpts,
              style(
                { fillColor: planColors.fgForDarkBG.hex },
                group([
                  headerSection,
                  gap(gapBelowHeaderSection),
                  style(
                    { strokeColor: planColors.fgForDarkBG.hex },
                    hr(hrLineHeight),
                  ),
                  block(() => {
                    switch (data.type) {
                      case 'range': {
                        return _getRangeSection(
                          dataX,
                          components,
                          rangeIds,
                          data,
                        )
                      }
                      case 'breakdown': {
                        return _getBreakdownSection(
                          dataX,
                          hoverTransitionMap,
                          components,
                          breakdownIds,
                          data,
                          {
                            height:
                              maxSize.height -
                              headerHeight -
                              gapBelowHeaderSection -
                              hrLineHeight -
                              boxOpts.padding.vert * 2,
                            width: maxSize.width - boxOpts.padding.horz * 2,
                          },
                          planColors,
                        )
                      }
                      default:
                        noCase(data)
                    }
                  }),
                ]),
              ),
            ),
          )
        },
      }
    },
  )

const fontSize = { large: { nonMono: 15, mono: 14 }, small: 12 }
// ----------------
// ---- Header ----
// ----------------
const _getHeaderSection = (
  dataX: number,
  components: ChartPointerBoxComponents,
  data: PlanResultsChartData,
) => {
  const { text, style, getLabel, oneRowGrid, fixedWidth, grid, group, gap } =
    components
  const { planParamsExt, planParams } = data
  const {
    asMFN,
    months,
    getCurrentAgeOfPerson,
    numMonths,
    monthsFromNowToCalendarMonth,
    getZonedTime,
  } = planParamsExt
  const calendarMonth = monthsFromNowToCalendarMonth(dataX)
  const labelBase = getLabel({ lineHeight: fontSize.large.nonMono })
  const getAgeBreakdown = (person: 'person1' | 'person2') => {
    if (dataX > asMFN(months[person].max)) {
      return {
        years: '—',
        months: '—',
      }
    }
    const inMonths = getCurrentAgeOfPerson(person).inMonths + dataX
    return {
      years: `${Math.floor(inMonths / 12)}`,
      months: `${inMonths % 12}`,
    }
  }

  const getAgeLine = (label: string, person: 'person1' | 'person2') => {
    const getPart = (n: string, l: string, width: number) =>
      fixedWidth(
        { width, align: 'end' },
        labelBase({}, [
          text({}, n),
          style(
            {
              font: ChartUtils.getFont(fontSize.small, '400'),
              opacity: 0.6,
            },
            text({ padding: { left: 3 } }, l),
          ),
        ]),
      )

    return [
      labelBase({}, text({}, label)),
      oneRowGrid({ gap: 0, align: { horz: 'end', vert: 'center' } }, [
        getPart(getAgeBreakdown(person).years, 'yr', 45),
        getPart(getAgeBreakdown(person).months, 'mo', 42),
      ]),
    ]
  }

  return group([
    style(
      { font: ChartUtils.getFont(fontSize.large.nonMono, 'bold') },

      dataX === numMonths
        ? labelBase({}, text({}, 'Legacy'))
        : planParams.people.withPartner
          ? grid(
              {
                gap: { horz: 30, vert: 5 },
                align: { horz: 'between', vert: 'end' },
              },
              [
                getAgeLine('Your Age', 'person1'),
                getAgeLine(`Partner's Age`, 'person2'),
              ],
            )
          : oneRowGrid(
              { gap: 30, align: { horz: 'between', vert: 'end' } },
              getAgeLine('Age', 'person1'),
            ),
    ),
    gap(4),
    style(
      {
        // font: ChartUtils.getFont(fontSize.small),
        font: ChartUtils.getFont(fontSize.large.nonMono - 2),
        opacity: 1,
      },
      labelBase(
        { align: 'end' },
        text({}, getZonedTime.fromObject(calendarMonth).toFormat('LLL yyyy')),
      ),
    ),
  ])
}

// ---------------
// ---- Range ----
// ---------------
const _getRangeSection = (
  dataX: number,
  components: ChartPointerBoxComponents,
  rangeIds: ChartRange<unknown>['ids'],
  data: Extract<PlanResultsChartData, { type: 'range' }>,
) => {
  const { getLabel, group, text, grid, style, mark, gap, circle, oneRowGrid } =
    components
  const labelLarge = getLabel({ lineHeight: fontSize.large.nonMono })
  const labelSmall = getLabel({ lineHeight: fontSize.small })

  const { formatY, planColors } = data

  const dataYs = block(() => {
    const ys = SimpleRange.Closed.isIn(dataX, data.range.xRange)
      ? fGet(data.range.yRangeByX[dataX])
      : null
    const explanations = _getRangeExplanation(data.chartType)
    return ys
      ? [
          {
            id: rangeIds.end,
            y: ys.end,
            label: '95',
            explain: explanations?.['95'] ?? null,
          },
          {
            id: rangeIds.mid,
            y: ys.mid,
            label: '50',
            explain: explanations?.['50'] ?? null,
          },
          {
            id: rangeIds.start,
            y: ys.start,
            label: '5',
            explain: explanations?.['5'] ?? null,
          },
        ]
      : []
  })
  return group([
    gap(13),
    style(
      { font: ChartUtils.getFont(fontSize.large.nonMono) },
      grid(
        {
          gap: { horz: 10, vert: dataYs[0].explain ? 15 : 10 },
          align: { vert: 'start', horz: 'between' },
        },
        dataYs.map(({ id, label, y, explain }) => [
          oneRowGrid({ gap: 10, align: { horz: 'start', vert: 'start' } }, [
            style(
              { fillColor: planColors.shades.main[7].hex },
              circle({
                box: { width: 10, height: 10 },
                center: { x: 5, y: 5 },
                radius: 5,
              }),
            ),
            group([
              labelLarge({ id, align: 'start' }, [
                text({}, label),
                style(
                  { font: ChartUtils.getMonoFont(7) },
                  text({ yOffset: 5 }, 'th'),
                ),
                text({}, ' percentile'),
              ]),
              ...(explain
                ? [
                    gap(2),
                    style(
                      {
                        font: ChartUtils.getFont(fontSize.small),
                        opacity: 0.6,
                      },
                      labelSmall({ id, align: 'start' }, [text({}, explain)]),
                    ),
                  ]
                : []),
            ]),
          ]),
          style(
            { font: ChartUtils.getMonoFont(fontSize.large.mono) },
            labelLarge({ align: 'end' }, text({}, `${formatY(y)}`)),
          ),
        ]),
      ),
    ),
  ])
}

// -------------------
// ---- Breakdown ----
// -------------------
const _getBreakdownSection = (
  dataX: number,
  hoverTransitionMap: Map<string, Transition<0 | 1> | null>,
  components: ChartPointerBoxComponents,
  breakdownIds: ChartBreakdown<unknown>['ids'],
  data: Extract<PlanResultsChartData, { type: 'breakdown' }>,
  availableSize: Size,
  colors: PlanColors,
) => {
  const { shades } = colors
  const {
    text,
    style,
    gap,
    barForText,
    grid,
    getLabel,
    group,
    circle,
    oneRowGrid,
  } = components

  const { formatY } = data

  const height = {
    lineHeight: fontSize.large.nonMono + 13,
    barAboveTotal: 2,
    gap: {
      beforeTotalLine: 8,
      afterTotalLine: -2,
    },
  }
  const width = {
    legendCircleR: 6,
    gap: {
      legendLabel: 12,
      labelAmount: 10,
    },
  }

  const labelLarge = getLabel({ lineHeight: height.lineHeight })

  const { partsToShow, partsNotShownSummary, partsTotal } = block(() => {
    const parts = data.breakdown.parts
      .slice()
      .reverse()
      .map(({ id, label, data, chartColorIndex }) => ({
        dataId: breakdownIds.part(id),
        label,
        pattern: getChartBandColor(chartColorIndex).fillPattern,
        value: _isInRange(dataX, data.xRange)
          ? ({ inRange: true, value: fGet(data.yByX[dataX]) } as const)
          : ({ inRange: false } as const),
        show: false,
      }))

    let numIncomeLinesToDisplay = block(() => {
      const availableHeight =
        availableSize.height -
        height.lineHeight - // portfolio line.
        height.gap.beforeTotalLine -
        height.barAboveTotal -
        height.gap.afterTotalLine -
        height.lineHeight

      const maxNumIncomeLines = Math.max(
        1,
        Math.floor(availableHeight / height.lineHeight),
      )
      return maxNumIncomeLines < parts.length
        ? maxNumIncomeLines - 1
        : maxNumIncomeLines
    })

    const firstHover = parts.find(
      (x) => hoverTransitionMap.get(x.dataId)?.target === 1,
    )
    if (firstHover) {
      firstHover.show = true
      numIncomeLinesToDisplay--
    }

    parts.forEach((part) => {
      if (numIncomeLinesToDisplay > 0 && !part.show) {
        part.show = true
        numIncomeLinesToDisplay--
      }
    })

    const [partsToShow, partsNotToShow] = _.partition(parts, (x) => x.show)
    const partsNotShownSummary =
      partsNotToShow.length > 0
        ? {
            count: partsNotToShow.length,
            total: _.sum(
              partsNotToShow.map((x) => (x.value.inRange ? x.value.value : 0)),
            ),
          }
        : null
    return {
      partsToShow,
      partsNotShownSummary,
      partsTotal: _.sum(
        parts.map((x) => (x.value.inRange ? x.value.value : 0)),
      ),
    }
  })

  const { total, remaining } = _isInRange(dataX, data.breakdown.total.xRange)
    ? block(() => {
        const total = fGet(data.breakdown.total.yByX[dataX])
        const remaining = total - partsTotal
        const pattern = {
          bg: { color: shades.light[5].rgb, opacity: 1 },
          stroke: Stroke.get(shades.main[5].rgb, 1),
          gap: 3,
        }
        return {
          remaining:
            remaining >= 0
              ? {
                  id: breakdownIds.remaining,
                  label: 'From Portfolio',
                  pattern,
                  value: { inRange: true, value: remaining } as const,
                  hover: letIn(
                    hoverTransitionMap.get(breakdownIds.remaining),
                    (transition) => (transition ? interpolate(transition) : 0),
                  ),
                }
              : {
                  id: '',
                  label: 'From Portfolio',
                  pattern,
                  value: { inRange: true, value: remaining } as const,
                  hover: 0,
                },
          total: { value: total },
        }
      })
    : { total: null, remaining: null }

  const table = _.compact([
    remaining,
    ...partsToShow.map(({ dataId, label, value, pattern }) => ({
      id: value.inRange ? dataId : null,
      label: label ?? '<no label>',
      pattern,
      value,
      hover: letIn(hoverTransitionMap.get(dataId), (transition) =>
        transition ? interpolate(transition) : 0,
      ),
    })),
    partsNotShownSummary
      ? {
          id: null,
          pattern: null,
          label: `… ${partsNotShownSummary.count} more`,
          value: { inRange: true, value: partsNotShownSummary.total } as const,
          hover: 0,
        }
      : null,
  ])

  const _getAmountLabel = (item: (typeof table)[number]) =>
    style(
      { font: ChartUtils.getMonoFont(fontSize.large.mono) },
      labelLarge(
        { align: 'end' },
        text({}, item.value.inRange ? formatY(item.value.value) : '—'),
      ),
    )
  const maxLabelWidth = block(() => {
    const maxAmount = _.max(table.map((x) => _getAmountLabel(x)().width)) ?? 0
    return (
      availableSize.width -
      width.legendCircleR * 2 -
      width.gap.legendLabel -
      width.gap.labelAmount -
      maxAmount
    )
  })

  return group([
    style(
      { font: ChartUtils.getFont(fontSize.large.nonMono) },
      group([
        grid(
          {
            gap: { vert: 0, horz: 0 },
            align: { vert: 'end', horz: 'between' },
          },
          table.map((x) => {
            const mixColors = (pattern: ChartStyling.StripePattern) =>
              interpolate({
                from: pattern.bg.color,
                target: pattern.stroke.color,
                progress: 0.5,
              })

            return [
              oneRowGrid({ gap: 0, align: { horz: 'start', vert: 'end' } }, [
                x.pattern
                  ? style(
                      { fillColor: RGB.toHex(mixColors(x.pattern)) },
                      circle({
                        box: {
                          width: width.legendCircleR * 2,
                          height: width.legendCircleR * 2 + 4,
                        },
                        center: {
                          x: width.legendCircleR,
                          y: width.legendCircleR,
                        },
                        radius: width.legendCircleR,
                      }),
                    )
                  : gap(0, width.legendCircleR * 2),
                gap(0, width.gap.legendLabel),
                labelLarge(
                  {
                    id: x.id ?? undefined,
                    box: {
                      borderRadius: 3,
                      padding: { left: 5, right: 5, top: -4, bottom: 4 },
                      fill: { color: shades.main[9].rgb },
                      percent: x.hover,
                    },
                  },
                  text({ maxWidth: maxLabelWidth }, x.label),
                ),
              ]),
              gap(0, width.gap.legendLabel),
              _getAmountLabel(x),
            ]
          }),
        ),
        ...(total
          ? block(() => {
              const label = `${formatY(total.value)}`
              return [
                gap(height.gap.beforeTotalLine),
                style(
                  {
                    opacity: 1,
                    strokeColor: colors.fgForDarkBG.hex,
                    font: ChartUtils.getMonoFont(fontSize.large.mono),
                  },
                  barForText({
                    text: label,
                    height: height.barAboveTotal,
                    align: 'end',
                  }),
                ),
                gap(height.gap.afterTotalLine),
                labelLarge({ align: 'end' }, [
                  text({ padding: { right: 10 } }, 'Total Spending'),
                  style(
                    { font: ChartUtils.getMonoFont(fontSize.large.mono) },
                    text({}, label),
                  ),
                ]),
              ]
            })
          : []),
      ]),
    ),
  ])
}

const _getRangeExplanation = (chartType: PlanResultsChartType) => {
  const proportional = {
    '95': 'if the market does very well',
    '50': 'the most likely scenario',
    '5': 'if the market does very badly',
  }
  const usuallyProportional = {
    '95': 'usually if the market does well',
    '50': proportional['50'],
    '5': 'usually if the market does badly',
  }
  const inverse = {
    '95': proportional['5'],
    '50': proportional['50'],
    '5': proportional['95'],
  }
  const usuallyInverse = {
    '95': usuallyProportional['5'],
    '50': usuallyProportional['50'],
    '5': usuallyProportional['95'],
  }
  switch (chartType) {
    case 'spending-total':
    case 'spending-total-funding-sources-5':
    case 'spending-total-funding-sources-50':
    case 'spending-total-funding-sources-95':
    case 'portfolio':
    case 'spending-general':
      return proportional
    case 'withdrawal':
    case 'asset-allocation-savings-portfolio':
      return usuallyInverse
    case 'asset-allocation-total-portfolio':
      return usuallyProportional
    default: {
      if (
        isPlanResultsChartSpendingDiscretionaryType(chartType) ||
        isPlanResultsChartSpendingEssentialType(chartType)
      )
        return proportional
      noCase(chartType)
    }
  }
}
