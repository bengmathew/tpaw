import {
  linearFnFomPoints,
  linearFnFromPointAndSlope,
  PlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import {
  extendPlanParams,
  PlanParamsExt,
} from '../../../../TPAWSimulator/PlanParamsExt'
import { TPAWRunInWorkerByPercentileByMonthsFromNow } from '../../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { UseTPAWWorkerResult } from '../../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { assert, fGet, noCase } from '../../../../Utils/Utils'
import {
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
  planChartSpendingDiscretionaryTypeID,
  planChartSpendingEssentialTypeID,
  PlanChartType,
} from '../PlanChartType'

export type TPAWChartDataMain = {
  params: PlanParams
  paramsExt: PlanParamsExt
  label: string
  type: PlanChartType
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
  successRate: number
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
    params: curr.params,
    paramsExt: curr.paramsExt,
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
    successRate: curr.successRate,
  }
}

const _spendingMonths = ({
  params,
  asMFN,
  withdrawalStartMonth,
}: PlanParamsExt) => {
  const withdrawalStart = asMFN(withdrawalStartMonth)

  return params.dev.alwaysShowAllMonths
    ? 'allMonths'
    : [
        ...params.adjustmentsToSpending.extraSpending.essential,
        ...params.adjustmentsToSpending.extraSpending.discretionary,
      ].some((x) => asMFN(x.monthRange).start < withdrawalStart)
    ? ('allMonths' as const)
    : ('retirementMonths' as const)
}

export const tpawChartDataMain = (
  type: Exclude<PlanChartType, 'reward-risk-ratio-comparison'>,
  tpawResult: UseTPAWWorkerResult,
): TPAWChartDataMain => {
  const { params } = tpawResult.args
  const paramsExt = extendPlanParams(params.original)
  const hasLegacy =
    params.adjustmentsToSpending.tpawAndSPAW.legacy.total !== 0 ||
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling !== null
  const spendingYears = _spendingMonths(paramsExt)
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
        formatPercentage(1),
        spendingYears,
        0,
        'auto',
      )
    default:
      if (isPlanChartSpendingEssentialType(type)) {
        const id = planChartSpendingEssentialTypeID(type)
        assert(
          params.original.adjustmentsToSpending.extraSpending.essential.find(
            (x) => x.id === id,
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
      if (isPlanChartSpendingDiscretionaryType(type)) {
        const id = planChartSpendingDiscretionaryTypeID(type)
        assert(
          params.original.adjustmentsToSpending.extraSpending.discretionary.find(
            (x) => x.id === id,
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
  type: PlanChartType,
  tpawResult: UseTPAWWorkerResult,
  dataFn: (
    tpawResult: UseTPAWWorkerResult,
  ) => TPAWRunInWorkerByPercentileByMonthsFromNow,
  yFormat: (x: number) => string,
  monthRange: 'retirementMonths' | 'allMonths',
  monthEndDelta: number,
  yDisplayRangeIn: 'auto' | SimpleRange,
): TPAWChartDataMain => {
  const { args } = tpawResult
  const { params } = args
  const paramsExt = extendPlanParams(params.original)
  const { asMFN, withdrawalStartMonth, maxMaxAge } = paramsExt
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
    [args.percentileRange.start, 50, args.percentileRange.end],
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

  const successRate = 1 - tpawResult.percentageOfRunsWithInsufficientFunds
  return {
    params: params.original,
    paramsExt,
    type,
    label: type,
    months,
    percentiles,
    min,
    max,
    yFormat,
    yDisplayRange,
    successRate,
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
