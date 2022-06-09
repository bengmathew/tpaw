import _ from 'lodash'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'
import {TPAWRunInWorkerByPercentileByYearsFromNow} from '../../../../TPAWSimulator/Worker/TPAWRunInWorker'
import {UseTPAWWorkerResult} from '../../../../TPAWSimulator/Worker/UseTPAWWorker'
import {formatCurrency} from '../../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../../Utils/FormatPercentage'
import {
  linearFnFomPoints,
  linearFnFromPointAndSlope,
} from '../../../../Utils/LinearFn'
import {SimpleRange} from '../../../../Utils/SimpleRange'
import {assert, fGet, noCase} from '../../../../Utils/Utils'
import {SimulationInfo} from '../../../App/WithSimulation'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from '../ChartPanelType'

export type TPAWChartDataMain = {
  label: string
  percentiles: {
    data: (x: number) => number
    percentile: number
    isHighlighted: boolean
  }[]
  min: {x: number; y: number}
  max: {x: number; y: number}
  years: {
    displayRange: SimpleRange
    retirement: number
    max: number
    display: (yearsFromNow: number) => number
  }
  yDisplayRange: SimpleRange
  yFormat: (x: number) => string
}

export const tpawChartDataScaled = (
  curr: TPAWChartDataMain,
  targetYRange: SimpleRange
): TPAWChartDataMain => {
  const scaleY = linearFnFomPoints(
    curr.yDisplayRange.start,
    targetYRange.start,
    curr.yDisplayRange.end,
    targetYRange.end
  )
  return {
    label: `${curr.label} scaled.`,
    percentiles: curr.percentiles.map(({data, percentile, isHighlighted}) => ({
      data: x => scaleY(data(x)),
      percentile,
      isHighlighted,
    })),
    min: {x: curr.min.x, y: scaleY(curr.min.y)},
    max: {x: curr.max.x, y: scaleY(curr.max.y)},
    yDisplayRange: {
      start: scaleY(curr.yDisplayRange.start),
      end: scaleY(curr.yDisplayRange.end),
    },
    years: curr.years,
    yFormat: curr.yFormat,
  }
}

export const tpawChartDataMain = (
  type: ChartPanelType,
  tpawResult: UseTPAWWorkerResult,
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartDataMain => {
  const {params} = tpawResult.args
  const {asYFN, withdrawalStartYear} = extendTPAWParams(params.original)
  const {legacy, spendingCeiling, withdrawals} = params
  const hasLegacy = legacy.total !== 0 || spendingCeiling !== null
  const withdrawalStart = asYFN(withdrawalStartYear)
  const spendingYears = params.display.alwaysShowAllYears
    ? 'allYears'
    : [
        ...params.withdrawals.essential,
        ...params.withdrawals.discretionary,
      ].some(x => asYFN(x.yearRange).start < withdrawalStart)
    ? ('allYears' as const)
    : ('retirementYears' as const)
  switch (type) {
    case 'spending-total':
      return _data(
        type,
        tpawResult,
        x => x.savingsPortfolio.withdrawals.total,
        x => formatCurrency(x),
        spendingYears,
        0,
        [],
        highlightPercentiles,
        'auto'
      )
    case 'spending-general':
      return _data(
        type,
        tpawResult,
        x => x.savingsPortfolio.withdrawals.regular,
        x => formatCurrency(x),
        spendingYears,
        0,
        [],
        highlightPercentiles,
        'auto'
      )

    case 'portfolio':
      return _data(
        type,
        tpawResult,
        x =>
          _addYear(
            x.savingsPortfolio.start.balance,
            x.endingBalanceOfSavingsPortfolioByPercentile
          ),
        x => formatCurrency(x),
        'allYears',
        1,
        [],
        highlightPercentiles,
        'auto'
      )
    case 'glide-path':
      return _data(
        type,
        tpawResult,
        x => x.savingsPortfolio.afterWithdrawals.allocation.stocks,
        formatPercentage(0),
        'allYears',
        hasLegacy ? 0 : -1,
        [],
        highlightPercentiles,
        {start: 0, end: 1}
      )
    case 'withdrawal-rate':
      return _data(
        type,
        tpawResult,
        x => x.savingsPortfolio.withdrawals.fromSavingsPortfolioRate,
        formatPercentage(1),
        params.display.alwaysShowAllYears ? 'allYears' : 'retirementYears',
        0,
        [],
        highlightPercentiles,
        'auto'
      )
    default:
      if (isChartPanelSpendingEssentialType(type)) {
        const id = chartPanelSpendingEssentialTypeID(type)
        assert(params.withdrawals.essential.find(x => x.id === id))
        return _data(
          type,
          tpawResult,
          x => fGet(x.savingsPortfolio.withdrawals.essential.byId.get(id)),
          x => formatCurrency(x),
          spendingYears,
          0,
          [],
          highlightPercentiles,
          'auto'
        )
      }
      if (isChartPanelSpendingDiscretionaryType(type)) {
        const id = chartPanelSpendingDiscretionaryTypeID(type)
        assert(params.withdrawals.discretionary.find(x => x.id === id))
        return _data(
          type,
          tpawResult,
          x => fGet(x.savingsPortfolio.withdrawals.discretionary.byId.get(id)),
          x => formatCurrency(x),
          spendingYears,
          0,
          [],
          highlightPercentiles,
          'auto'
        )
      }
      noCase(type)
  }
}

const _addYear = (
  {byPercentileByYearsFromNow}: ReturnType<Parameters<typeof _data>[2]>,
  numberByPercentile: {data: number; percentile: number}[]
) => ({
  byPercentileByYearsFromNow: byPercentileByYearsFromNow.map(
    ({data, percentile}, p) => ({
      data: [...data, numberByPercentile[p].data],
      percentile,
    })
  ),
})

const _data = (
  label: string,
  tpawResult: UseTPAWWorkerResult,
  dataFn: (
    tpawResult: UseTPAWWorkerResult
  ) => TPAWRunInWorkerByPercentileByYearsFromNow,
  yFormat: (x: number) => string,
  yearRange: 'retirementYears' | 'allYears',
  yearEndDelta: number,
  ageGroups: SimpleRange[],
  highlightPercentiles: number[],
  yDisplayRangeIn: 'auto' | SimpleRange
): TPAWChartDataMain => {
  const {args} = tpawResult
  const {params} = args
  const {asYFN, withdrawalStartYear, pickPerson, maxMaxAge} = extendTPAWParams(
    params.original
  )
  const retirement = asYFN(withdrawalStartYear)
  const maxYear = asYFN(maxMaxAge)
  const xAxisPerson = params.people.withPartner
    ? pickPerson(params.people.xAxis)
    : params.people.person1
  const years: TPAWChartDataMain['years'] = {
    displayRange: {
      start: yearRange === 'retirementYears' ? retirement : 0,
      end: maxYear + yearEndDelta,
    },
    retirement,
    max: maxYear,
    display: yearsFromNow => yearsFromNow + xAxisPerson.ages.current,
  }
  const {byPercentileByYearsFromNow} = dataFn(tpawResult)

  const percentiles = _addPercentileInfo(
    _interpolate(byPercentileByYearsFromNow, years.displayRange),
    args.percentiles,
    highlightPercentiles
  )

  const last = fGet(_.last(byPercentileByYearsFromNow)).data
  const first = fGet(_.first(byPercentileByYearsFromNow)).data
  const maxY = Math.max(...last)
  const minY = Math.min(...first)

  const max = {x: last.indexOf(maxY), y: maxY}
  const min = {x: first.indexOf(minY), y: minY}
  const yDisplayRange =
    yDisplayRangeIn === 'auto'
      ? {start: Math.min(0, min.y), end: Math.max(0.0001, max.y)}
      : yDisplayRangeIn
  return {label, years, percentiles, min, max, yFormat, yDisplayRange}
}

const _addPercentileInfo = <T>(
  ys: T[],
  percentiles: number[],
  highlightedPercentiles: number[]
) => {
  return ys.map((data, i) => ({
    data,
    percentile: percentiles[i],
    isHighlighted: highlightedPercentiles.includes(percentiles[i]),
  }))
}

const _interpolate = (
  ys: {data: number[]}[],
  {start: xStart, end: xEnd}: SimpleRange
) => {
  const beforeSlope = _avgSlope(ys, xStart)
  const afterSlope = _avgSlope(ys, xEnd)
  return ys.map(({data}) => {
    const extrapolateBefore = linearFnFromPointAndSlope(
      xStart,
      data[xStart],
      beforeSlope
    )
    const extrapolateAfter = linearFnFromPointAndSlope(
      xEnd,
      data[xEnd],
      afterSlope
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

const _avgSlope = (ys: {data: number[]}[], i: number) => {
  const iPlus1 = Math.min(i + 1, ys[0].data.length - 1)
  const sum = _.sum(ys.map(x => x.data[iPlus1] - x.data[i]))
  const result = sum / ys.length
  return result
}
