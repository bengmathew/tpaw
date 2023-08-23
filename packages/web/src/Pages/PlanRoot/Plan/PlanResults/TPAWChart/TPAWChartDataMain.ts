import {
  NonPlanParams,
  PlanParams,
  linearFnFomPoints,
  linearFnFromPointAndSlope,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsExtended } from '../../../../../TPAWSimulator/ExtentPlanParams'
import { TPAWRunInWorkerByPercentileByMonthsFromNow } from '../../../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { UseTPAWWorkerResult } from '../../../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatCurrency } from '../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { SimpleRange } from '../../../../../Utils/SimpleRange'
import { fGet, noCase } from '../../../../../Utils/Utils'
import { optGet } from '../../../../../Utils/optGet'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  planResultsChartSpendingDiscretionaryTypeID,
  planResultsChartSpendingEssentialTypeID,
} from '../PlanResultsChartType'

export type TPAWChartDataMain = {
  planParams: PlanParams
  planParamsExt: PlanParamsExtended
  nonPlanParams: NonPlanParams
  label: string
  type: PlanResultsChartType
  percentiles: {
    data: (x: number) => number
    percentile: number
  }[]

  min: { x: number; y: number }
  max: { x: number; y: number }
  months: {
    displayRange: SimpleRange
    retirement: number
    max: number
  }
  yDisplayRange: SimpleRange
  yFormat: (x: number) => string
}

export const tpawChartDataScaled = (
  curr: TPAWChartDataMain,
  targetYRange: SimpleRange,
): TPAWChartDataMain => {
  const scaleY = linearFnFomPoints(
    curr.yDisplayRange.start,
    targetYRange.start,
    curr.yDisplayRange.end,
    targetYRange.end,
  )

  return {
    planParams: curr.planParams,
    nonPlanParams: curr.nonPlanParams,
    planParamsExt: curr.planParamsExt,
    label: `${curr.label} scaled.`,
    type: curr.type,
    percentiles: curr.percentiles.map(({ data, percentile }) => ({
      data: (x: number) => scaleY(data(x)),
      percentile,
    })),
    min: { x: curr.min.x, y: scaleY(curr.min.y) },
    max: { x: curr.max.x, y: scaleY(curr.max.y) },
    yDisplayRange: {
      start: scaleY(curr.yDisplayRange.start),
      end: scaleY(curr.yDisplayRange.end),
    },
    months: curr.months,
    yFormat: curr.yFormat,
  }
}

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

export const tpawChartDataMain = (
  type: PlanResultsChartType,
  tpawResult: UseTPAWWorkerResult,
): TPAWChartDataMain => {
  const { params, planParamsExt, nonPlanParams } = tpawResult
  const hasLegacy =
    params.adjustmentsToSpending.tpawAndSPAW.legacy.total !== 0 ||
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling !== null
  const spendingYears = _spendingMonths(planParamsExt, nonPlanParams)
  switch (type) {
    case 'spending-total':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) => x.savingsPortfolio.withdrawals.total,
        (x) => formatCurrency(x),
        spendingYears,
        0,
        'auto',
      )
    case 'spending-general':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) => x.savingsPortfolio.withdrawals.regular,
        (x) => formatCurrency(x),
        spendingYears,
        0,
        'auto',
      )

    case 'portfolio':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) =>
          _addMonth(
            x.savingsPortfolio.start.balance,
            x.endingBalanceOfSavingsPortfolioByPercentile,
          ),
        (x) => formatCurrency(x),
        'allMonths',
        1,
        'auto',
      )
    case 'asset-allocation-savings-portfolio':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) => x.savingsPortfolio.afterWithdrawals.allocation.stocks,
        formatPercentage(0),
        'allMonths',
        hasLegacy ? 0 : -1,
        { start: 0, end: 1 },
      )
    case 'asset-allocation-total-portfolio':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) => x.totalPortfolio.afterWithdrawals.allocation.stocks,
        formatPercentage(0),
        'allMonths',
        hasLegacy ? 0 : -1,
        { start: 0, end: 1 },
      )
    case 'withdrawal':
      return _dataPercentiles(
        type,
        tpawResult,
        (x) => x.savingsPortfolio.withdrawals.fromSavingsPortfolioRate,
        formatPercentage(2),
        spendingYears,
        0,
        'auto',
      )
    default:
      if (isPlanResultsChartSpendingEssentialType(type)) {
        const id = planResultsChartSpendingEssentialTypeID(type)
        fGet(
          optGet(
            params.original.adjustmentsToSpending.extraSpending.essential,
            id,
          ),
        )
        return _dataPercentiles(
          type,
          tpawResult,
          (x) => fGet(x.savingsPortfolio.withdrawals.essential.byId.get(id)),
          (x) => formatCurrency(x),
          spendingYears,
          0,
          'auto',
        )
      }
      if (isPlanResultsChartSpendingDiscretionaryType(type)) {
        const id = planResultsChartSpendingDiscretionaryTypeID(type)
        fGet(
          optGet(
            params.original.adjustmentsToSpending.extraSpending.discretionary,
            id,
          ),
        )
        return _dataPercentiles(
          type,
          tpawResult,
          (x) =>
            fGet(x.savingsPortfolio.withdrawals.discretionary.byId.get(id)),
          (x) => formatCurrency(x),
          spendingYears,
          0,
          'auto',
        )
      }
      noCase(type)
  }
}

const _addMonth = (
  {
    byPercentileByMonthsFromNow,
  }: ReturnType<Parameters<typeof _dataPercentiles>[2]>,
  numberByPercentile: { data: number; percentile: number }[],
) => ({
  byPercentileByMonthsFromNow: byPercentileByMonthsFromNow.map(
    ({ data, percentile }, p) => ({
      data: [...data, numberByPercentile[p].data],
      percentile,
    }),
  ),
})

const _dataPercentiles = (
  type: PlanResultsChartType,
  tpawResult: UseTPAWWorkerResult,
  dataFn: (
    tpawResult: UseTPAWWorkerResult,
  ) => TPAWRunInWorkerByPercentileByMonthsFromNow,
  yFormat: (x: number) => string,
  monthRange: 'retirementMonths' | 'allMonths',
  monthEndDelta: number,
  yDisplayRangeIn: 'auto' | SimpleRange,
): TPAWChartDataMain => {
  const { params, planParamsExt, nonPlanParams } = tpawResult
  const { asMFN, withdrawalStartMonth, maxMaxAge } = planParamsExt

  const retirement = asMFN(withdrawalStartMonth)

  const maxMonth = asMFN(maxMaxAge)
  const months: TPAWChartDataMain['months'] = {
    displayRange: {
      start: monthRange === 'retirementMonths' ? retirement : 0,
      end: maxMonth + monthEndDelta,
    },
    retirement,
    max: maxMonth,
  }
  const { byPercentileByMonthsFromNow } = dataFn(tpawResult)

  const percentiles = _addPercentileInfo(
    _interpolate(byPercentileByMonthsFromNow, months.displayRange),
    [
      nonPlanParams.percentileRange.start,
      50,
      nonPlanParams.percentileRange.end,
    ],
  )

  const last = fGet(_.last(byPercentileByMonthsFromNow)).data
  const first = fGet(_.first(byPercentileByMonthsFromNow)).data
  const maxY = Math.max(...last)
  const minY = Math.min(...first)

  const max = { x: last.indexOf(maxY), y: maxY }
  const min = { x: first.indexOf(minY), y: minY }
  const yDisplayRange =
    yDisplayRangeIn === 'auto'
      ? { start: Math.min(0, min.y), end: Math.max(0.0001, max.y) }
      : yDisplayRangeIn

  return {
    planParams: params.original,
    nonPlanParams,
    planParamsExt,
    type,
    label: type,
    months,
    percentiles,
    min,
    max,
    yFormat,
    yDisplayRange,
  }
}

const _addPercentileInfo = <T>(ys: T[], percentiles: number[]) => {
  return ys.map((data, i) => ({
    data,
    percentile: percentiles[i],
  }))
}

const _interpolate = (
  ys: { data: number[] }[],
  { start: xStart, end: xEnd }: SimpleRange,
) => {
  const beforeSlope = _avgSlope(ys, xStart)
  const afterSlope = _avgSlope(ys, xEnd)
  return ys.map(({ data }) => {
    const extrapolateBefore = linearFnFromPointAndSlope(
      xStart,
      data[xStart],
      beforeSlope,
    )
    const extrapolateAfter = linearFnFromPointAndSlope(
      xEnd,
      data[xEnd],
      afterSlope,
    )
    const interpolated = data
      .slice(0, -1)
      .map((v, i) => linearFnFomPoints(i, v, i + 1, data[i + 1]))
    const dataFn = (a: number) => {
      if (a <= xStart) return extrapolateBefore(a)
      if (a >= xEnd) return extrapolateAfter(a)
      const section = Math.floor(a)
      const currFn = interpolated[section]
      const result = currFn(a)
      return result
    }
    dataFn.data = data
    return dataFn
  })
}

const _avgSlope = (ys: { data: number[] }[], i: number) => {
  const iPlus1 = Math.min(i + 1, ys[0].data.length - 1)
  const sum = _.sum(ys.map((x) => x.data[iPlus1] - x.data[i]))
  const result = sum / ys.length
  return result
}
