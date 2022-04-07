import _ from 'lodash'
import {TPAWParams, ValueForYearRange} from '../../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../../../TPAWSimulator/TPAWParamsProcessed'
import {TPAWRunInWorkerByPercentileByYearsFromNow} from '../../../TPAWSimulator/Worker/TPAWRunInWorker'
import {UseTPAWWorkerResult} from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import {
  linearFnFomPoints,
  linearFnFromPointAndSlope,
} from '../../../Utils/LinearFn'
import {nominalToReal} from '../../../Utils/NominalToReal'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {assert, fGet, noCase} from '../../../Utils/Utils'
import {SimulationInfo} from '../../App/WithSimulation'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from './ChartPanelType'

export type TPAWChartData = {
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

  isAgeInGroup: (age: number) => boolean
}

export const tpawChartDataYRange = ({min, max}: TPAWChartData) => ({
  start: Math.min(0, min.y),
  end: Math.max(0.0001, max.y),
})

export const tpawChartDataScaled = (
  curr: TPAWChartData,
  targetYRange: SimpleRange
): TPAWChartData => {
  const currYRange = tpawChartDataYRange(curr)
  const scaleY = linearFnFomPoints(
    currYRange.start,
    targetYRange.start,
    currYRange.end,
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
    years: curr.years,
    isAgeInGroup: curr.isAgeInGroup,
  }
}

export const tpawChartData = (
  type: ChartPanelType,
  tpawResult: SimulationInfo['tpawResult'],
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartData => {
  const {params} = tpawResult.args
  const {asYFN, withdrawalStartYear} = extendTPAWParams(params)
  const {legacy, spendingCeiling, withdrawals} = params
  const hasLegacy = legacy.total !== 0 || spendingCeiling !== null
  const withdrawalStart = asYFN(withdrawalStartYear)
  const spendingYears = [
    ...params.withdrawals.fundedByBonds,
    ...params.withdrawals.fundedByRiskPortfolio,
  ].some(x => asYFN(x.yearRange).start < withdrawalStart)
    ? ('allYears' as const)
    : ('retirementYears' as const)
  switch (type) {
    case 'spending-total':
      return _data(
        type,
        tpawResult,
        x => x.withdrawals.total,
        spendingYears,
        0,
        [],
        highlightPercentiles
      )
    case 'spending-general':
      return _data(
        type,
        tpawResult,
        x => x.withdrawals.regular,
        spendingYears,
        0,
        [],
        highlightPercentiles
      )

    case 'portfolio':
      return _data(
        type,
        tpawResult,
        x =>
          _addYear(
            x.startingBalanceOfSavingsPortfolio,
            x.endingBalanceOfSavingsPortfolioByPercentile
          ),
        'allYears',
        1,
        [],
        highlightPercentiles
      )
    case 'glide-path':
      return _data(
        type,
        tpawResult,
        x => x.savingsPortfolioStockAllocation,
        'allYears',
        hasLegacy ? 0 : -1,
        [],
        highlightPercentiles
      )
    case 'withdrawal-rate':
      return _data(
        type,
        tpawResult,
        x => x.withdrawalFromSavingsRate,
        'retirementYears',
        0,
        [],
        highlightPercentiles
      )
    default:
      if (isChartPanelSpendingEssentialType(type)) {
        const id = chartPanelSpendingEssentialTypeID(type)
        const index = params.withdrawals.fundedByBonds.findIndex(
          x => x.id === id
        )
        assert(index !== -1)
        return _data(
          type,
          tpawResult,
          x =>
            _separateExtraWithdrawal(
              params.withdrawals.fundedByBonds[index],
              params,
              x.withdrawals.essential,
              'essential'
            ),
          spendingYears,
          0,
          [],
          highlightPercentiles
        )
      }
      if (isChartPanelSpendingDiscretionaryType(type)) {
        const id = chartPanelSpendingDiscretionaryTypeID(type)
        const index = params.withdrawals.fundedByRiskPortfolio.findIndex(
          x => x.id === id
        )
        assert(index !== -1)
        return _data(
          type,
          tpawResult,
          x =>
            _separateExtraWithdrawal(
              params.withdrawals.fundedByRiskPortfolio[index],
              params,
              x.withdrawals.extra,
              'extra'
            ),
          spendingYears,
          0,
          [],
          highlightPercentiles
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

const _separateExtraWithdrawal = (
  extraWithdrawal: ValueForYearRange,
  params: TPAWParams,
  x: TPAWRunInWorkerByPercentileByYearsFromNow,
  type: 'extra' | 'essential'
): TPAWRunInWorkerByPercentileByYearsFromNow => {
  const processedParams = processTPAWParams(params)
  const yearRange = extendTPAWParams(params).asYFN(extraWithdrawal.yearRange)

  const byPercentileByYearsFromNow = x.byPercentileByYearsFromNow.map(
    ({data, percentile}) => ({
      data: data.map((value, yearFromNow) => {
        if (yearFromNow < yearRange.start || yearFromNow > yearRange.end) {
          return 0
        }
        const currYearParams = processedParams.byYear[yearFromNow].withdrawals
        const withdrawalTargetForThisYear = nominalToReal(
          extraWithdrawal,
          params.inflation,
          yearFromNow
        )
        if (withdrawalTargetForThisYear === 0) return 0
        const ratio =
          withdrawalTargetForThisYear /
          (type === 'extra'
            ? currYearParams.fundedByRiskPortfolio
            : currYearParams.fundedByBonds)
        assert(!isNaN(ratio)) // withdrawalTargetForThisYear ?>0 imples denominator is not 0.
        return value * ratio
      }),
      percentile,
    })
  )
  return {byPercentileByYearsFromNow}
}

const _data = (
  label: string,
  tpawResult: UseTPAWWorkerResult,
  dataFn: (
    tpawResult: UseTPAWWorkerResult
  ) => TPAWRunInWorkerByPercentileByYearsFromNow,
  yearRange: 'retirementYears' | 'allYears',
  yearEndDelta: number,
  ageGroups: SimpleRange[],
  highlightPercentiles: number[]
): TPAWChartData => {
  const {args} = tpawResult
  const {params} = args
  const {asYFN, withdrawalStartYear, pickPerson, maxMaxAge} =
    extendTPAWParams(params)
  const retirement = asYFN(withdrawalStartYear)
  const maxYear = asYFN(maxMaxAge)
  const xAxisPerson = params.people.withPartner
    ? pickPerson(params.people.xAxis)
    : params.people.person1
  const years: TPAWChartData['years'] = {
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

  const isAgeInGroup = (age: number) =>
    ageGroups.some(x => x.start <= age && x.end >= age)
  return {label, years, percentiles, min, max, isAgeInGroup}
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
      .map((v, i) =>
        linearFnFomPoints(i , v, i + 1, data[i + 1])
      )
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
