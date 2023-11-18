import { NonPlanParams, PlanParams, block, noCase } from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsExtended } from '../../../../../../TPAWSimulator/ExtentPlanParams'
import { PlanParamsProcessed } from '../../../../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { TPAWRunInWorkerByPercentileByMonthsFromNow } from '../../../../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { UseTPAWWorkerResult } from '../../../../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatCurrency } from '../../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../../Utils/FormatPercentage'
import { XY } from '../../../../../../Utils/Geometry'
import { SimpleRange } from '../../../../../../Utils/SimpleRange'
import { fGet } from '../../../../../../Utils/Utils'
import { optGet } from '../../../../../../Utils/optGet'
import { PlanSizing } from '../../../PlanSizing/PlanSizing'
import { PlanTransitionState } from '../../../PlanTransition'
import { PlanColors } from '../../../UsePlanColors'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  isPlanResultsChartSpendingTotalFundingSourcesType,
  planResultsChartSpendingDiscretionaryTypeID,
  planResultsChartSpendingEssentialTypeID,
  getPlanResultsChartSpendingTotalFundingSourcesPercentile,
} from '../../PlanResultsChartType'

export type PlanResultsChartData = {
  chartType: PlanResultsChartType
  planParams: PlanParams
  planParamsProcessed: PlanParamsProcessed
  planParamsExt: PlanParamsExtended
  nonPlanParams: NonPlanParams
  displayRange: { x: SimpleRange; y: SimpleRange }
  formatY: (x: number) => string
  planSizing: PlanSizing
  planTransitionState: PlanTransitionState
  planColors: PlanColors
} & (
  | {
      type: 'range'
      range: {
        yRangeByX: ((SimpleRange & { mid: number }) | null)[]
        xRange: SimpleRange
        minMax: { min: XY; max: XY }
      }
    }
  | {
      type: 'breakdown'
      breakdown: {
        range: {
          yRangeByX: ((SimpleRange & { mid: number }) | null)[]
          xRange: SimpleRange
          minMax: { min: XY; max: XY }
        }
        total: {
          yByX: (number | null)[] | Float64Array
          xRange: SimpleRange | null
          minMax: { min: XY; max: XY }
        }
        parts: {
          id: string
          chartColorIndex: number
          label: string | null
          data: {
            yByX: (number | null)[] | Float64Array
            xRange: SimpleRange | null
          }
        }[]
      }
    }
)

const _spendingMonths = (
  { planParams, asMFN, withdrawalStartMonth }: PlanParamsExtended,
  nonPlanParams: NonPlanParams,
) => {
  const withdrawalStart = asMFN(withdrawalStartMonth)

  return nonPlanParams.dev.alwaysShowAllMonths
    ? 'allMonths'
    : [
        ..._.values(planParams.adjustmentsToSpending.extraSpending.essential),
        ..._.values(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
        ),
      ].some((x) => asMFN(x.monthRange).start < withdrawalStart)
    ? ('allMonths' as const)
    : ('retirementMonths' as const)
}

export const getPlanResultsChartData = (
  chartType: PlanResultsChartType,
  tpawResult: UseTPAWWorkerResult,
  planSizing: PlanSizing,
  planTransitionState: PlanTransitionState,
  planColors: PlanColors,
): PlanResultsChartData => {
  const {
    params: planParamsProcessed,
    planParamsExt,
    nonPlanParams,
  } = tpawResult
  const planParams = planParamsProcessed.original
  const hasLegacy =
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.total !== 0 ||
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW
      .monthlySpendingCeiling !== null

  const xRangeAsMFN = block(() => {
    const { asMFN, withdrawalStartMonth, maxMaxAge } = planParamsExt
    const retirementStartAsMFN = nonPlanParams.dev.alwaysShowAllMonths
      ? 0
      : asMFN(withdrawalStartMonth)
    const lastPlanningMonthAsMFN = asMFN(maxMaxAge)
    return {
      allMonthsButLast: { start: 0, end: lastPlanningMonthAsMFN - 1 },
      allMonthsNoLegacy: { start: 0, end: lastPlanningMonthAsMFN },
      allMonthsWithLegacy: { start: 0, end: lastPlanningMonthAsMFN + 1 },
      retirementMonthsNoLegacy: {
        start: retirementStartAsMFN,
        end: lastPlanningMonthAsMFN,
      },
      retirementMonthsWithLegacy: {
        start: retirementStartAsMFN,
        end: lastPlanningMonthAsMFN + 1,
      },
    }
  })

  const spendingMonthsAsMFN = block(() => {
    const { asMFN, withdrawalStartMonth } = planParamsExt
    const withdrawalStartAsMFN = asMFN(withdrawalStartMonth)
    return [
      ..._.values(planParams.adjustmentsToSpending.extraSpending.essential),
      ..._.values(planParams.adjustmentsToSpending.extraSpending.discretionary),
    ].some((x) => asMFN(x.monthRange).start < withdrawalStartAsMFN)
      ? xRangeAsMFN.allMonthsNoLegacy
      : xRangeAsMFN.retirementMonthsNoLegacy
  })

  const _getPercentile = (
    { byPercentileByMonthsFromNow }: TPAWRunInWorkerByPercentileByMonthsFromNow,
    percentile: number,
  ) =>
    fGet(byPercentileByMonthsFromNow.find((x) => x.percentile === percentile))
      .data

  const common = {
    chartType,
    planParams,
    planParamsProcessed,
    planParamsExt,
    nonPlanParams,
    planSizing,
    planTransitionState,
    planColors,
  }

  const _processRange = (x: {
    range: TPAWRunInWorkerByPercentileByMonthsFromNow
    mfnRange: SimpleRange
    formatY: (x: number) => string
    minYDisplayRangeEnd: number
  }): Extract<PlanResultsChartData, { type: 'range' }> => {
    const { range, mfnRange, formatY, minYDisplayRangeEnd } = x
    const percentiles = {
      start: _getPercentile(range, 5),
      mid: _getPercentile(range, 50),
      end: _getPercentile(range, 95),
    }
    const minMax = block(() => {
      const xs = _.range(mfnRange.start, mfnRange.end + 1)
      let min = { x: 0, y: Infinity }
      let max = { x: 0, y: -Infinity }
      xs.forEach((x) => {
        if (percentiles.start[x] < min.y) min = { x, y: percentiles.start[x] }
        if (percentiles.end[x] > max.y) max = { x, y: percentiles.end[x] }
      })
      return { min, max }
    })
    return {
      ...common,
      displayRange: {
        x: mfnRange,
        y: { start: 0, end: Math.max(minYDisplayRangeEnd, minMax.max.y) },
      },
      formatY: formatY,
      type: 'range',
      range: {
        yRangeByX: _.zipWith(
          percentiles.start,
          percentiles.mid,
          percentiles.end,
          (start, mid, end) => ({ start, mid, end }),
        ),
        xRange: mfnRange,
        minMax: minMax,
      },
    }
  }

  const _processBreakdown = (x: {
    mfnRange: SimpleRange
    formatY: (x: number) => string
    range: TPAWRunInWorkerByPercentileByMonthsFromNow
    total: number[] | Float64Array
    parts: {
      id: string
      colorIndex: number
      sortIndex: number
      label: string | null
      yByX: number[] | Float64Array
      xRange: SimpleRange | null
    }[]
    minYDisplayRangeEnd: number
  }): PlanResultsChartData => {
    const { mfnRange, parts, formatY, total, range, minYDisplayRangeEnd } = x
    const dataXs = _.range(mfnRange.start, mfnRange.end + 1)

    const minMaxOfTotal = block(() => {
      let min = { x: 0, y: Infinity }
      let max = { x: 0, y: -Infinity }
      dataXs.forEach((dataX) => {
        const y = total[dataX]
        if (y < min.y) min = { x: dataX, y }
        if (y > max.y) max = { x: dataX, y }
      })
      return { min, max }
    })

    const { range: processedRange, displayRange } = _processRange({
      range,
      mfnRange,
      formatY,
      minYDisplayRangeEnd,
    })

    return {
      ...common,
      type: 'breakdown',
      displayRange,
      formatY,
      breakdown: {
        range: processedRange,
        total: { yByX: total, xRange: mfnRange, minMax: minMaxOfTotal },
        parts: parts
          .slice()
          .sort((a, b) => {
            if (_.isEqual(a.xRange, b.xRange)) return a.sortIndex - b.sortIndex
            if (!a.xRange) return 1
            if (!b.xRange) return -1
            if (
              a.xRange.start <= b.xRange.start &&
              a.xRange.end >= b.xRange.end
            )
              return -1
            if (
              b.xRange.start <= a.xRange.start &&
              b.xRange.end >= a.xRange.end
            )
              return 1
            return a.sortIndex - b.sortIndex
          })
          .map(({ id, colorIndex, yByX, xRange, label }) => ({
            id,
            label,
            chartColorIndex: colorIndex,
            data: { yByX, xRange },
          })),
      },
    }
  }

  switch (chartType) {
    case 'spending-total':
      return _processRange({
        range: tpawResult.savingsPortfolio.withdrawals.total,
        mfnRange: spendingMonthsAsMFN,
        formatY: (y) => formatCurrency(y),
        minYDisplayRangeEnd: 10,
      })

    case 'spending-general':
      return _processRange({
        range: tpawResult.savingsPortfolio.withdrawals.regular,
        mfnRange: xRangeAsMFN.retirementMonthsNoLegacy,
        formatY: (y) => formatCurrency(y),
        minYDisplayRangeEnd: 10,
      })

    case 'portfolio':
      return _processRange({
        range: _addMonth(
          tpawResult.savingsPortfolio.start.balance,
          tpawResult.endingBalanceOfSavingsPortfolioByPercentile,
        ),
        mfnRange: xRangeAsMFN.allMonthsWithLegacy,
        formatY: (y) => formatCurrency(y),
        minYDisplayRangeEnd: 10,
      })
    case 'asset-allocation-savings-portfolio':
      return _processRange({
        range: tpawResult.savingsPortfolio.afterWithdrawals.allocation.stocks,
        mfnRange: hasLegacy
          ? xRangeAsMFN.allMonthsNoLegacy
          : xRangeAsMFN.allMonthsButLast,
        formatY: formatPercentage(0),
        minYDisplayRangeEnd: 1,
      })
    case 'asset-allocation-total-portfolio':
      return _processRange({
        range: tpawResult.totalPortfolio.afterWithdrawals.allocation.stocks,
        mfnRange: hasLegacy
          ? xRangeAsMFN.allMonthsNoLegacy
          : xRangeAsMFN.allMonthsButLast,
        formatY: formatPercentage(0),
        minYDisplayRangeEnd: 1,
      })
    case 'withdrawal':
      return _processRange({
        range: tpawResult.savingsPortfolio.withdrawals.fromSavingsPortfolioRate,
        mfnRange: xRangeAsMFN.retirementMonthsNoLegacy,
        formatY: formatPercentage(2),
        minYDisplayRangeEnd: 1,
      })
    default:
      if (isPlanResultsChartSpendingTotalFundingSourcesType(chartType)) {
        const percentile = parseInt(
          getPlanResultsChartSpendingTotalFundingSourcesPercentile(chartType),
        )
        return _processBreakdown({
          mfnRange: spendingMonthsAsMFN,
          formatY: (y) => formatCurrency(y),
          range: tpawResult.savingsPortfolio.withdrawals.total,
          total: fGet(
            tpawResult.savingsPortfolio.withdrawals.total.byPercentileByMonthsFromNow.find(
              (x) => x.percentile === percentile,
            ),
          ).data,
          parts: [
            ..._.toPairs(
              planParamsProcessed.byMonth.wealth.futureSavings.byId,
            ).map(([id, x]) => ({
              yByX: x.values,
              xRange: x.validRange,
              id: `futureSavings-${id}`,
              label: planParams.wealth.futureSavings[id].label,
              sortIndex: planParams.wealth.futureSavings[id].sortIndex,
              colorIndex: planParams.wealth.futureSavings[id].colorIndex,
            })),
            ..._.toPairs(
              planParamsProcessed.byMonth.wealth.incomeDuringRetirement.byId,
            ).map(([id, x]) => ({
              yByX: x.values,
              xRange: x.validRange,
              id: `incomeDuringRetirement-${id}`,
              label: planParams.wealth.incomeDuringRetirement[id].label,
              sortIndex: planParams.wealth.incomeDuringRetirement[id].sortIndex,
              colorIndex:
                planParams.wealth.incomeDuringRetirement[id].colorIndex,
            })),
          ],
          minYDisplayRangeEnd: 10,
        })
      }
      if (isPlanResultsChartSpendingEssentialType(chartType)) {
        const id = planResultsChartSpendingEssentialTypeID(chartType)
        fGet(
          optGet(planParams.adjustmentsToSpending.extraSpending.essential, id),
        )
        return _processRange({
          range: fGet(
            tpawResult.savingsPortfolio.withdrawals.essential.byId.get(id),
          ),
          mfnRange: spendingMonthsAsMFN,
          formatY: (y) => formatCurrency(y),
          minYDisplayRangeEnd: 10,
        })
      }
      if (isPlanResultsChartSpendingDiscretionaryType(chartType)) {
        const id = planResultsChartSpendingDiscretionaryTypeID(chartType)
        fGet(
          optGet(
            planParams.adjustmentsToSpending.extraSpending.discretionary,
            id,
          ),
        )
        return _processRange({
          range: fGet(
            tpawResult.savingsPortfolio.withdrawals.discretionary.byId.get(id),
          ),
          mfnRange: spendingMonthsAsMFN,
          formatY: (y) => formatCurrency(y),
          minYDisplayRangeEnd: 10,
        })
      }
      noCase(chartType)
  }
}

const _addMonth = (
  { byPercentileByMonthsFromNow }: TPAWRunInWorkerByPercentileByMonthsFromNow,
  numberByPercentile: { data: number; percentile: number }[],
) => ({
  byPercentileByMonthsFromNow: byPercentileByMonthsFromNow.map(
    ({ data, percentile }, p) => ({
      data: [...data, numberByPercentile[p].data],
      percentile,
    }),
  ),
})
