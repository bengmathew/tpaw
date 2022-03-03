import _ from 'lodash'
import { TPAWParams, ValueForYearRange } from '../../../TPAWSimulator/TPAWParams'
import {
  numericYear,
  numericYearRange,
  processTPAWParams
} from '../../../TPAWSimulator/TPAWParamsProcessed'
import { TPAWRunInWorkerByPercentileByYearsFromNow } from '../../../TPAWSimulator/Worker/TPAWRunInWorker'
import { UseTPAWWorkerResult } from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import {
  linearFnFomPoints,
  linearFnFromPointAndSlope
} from '../../../Utils/LinearFn'
import { nominalToReal } from '../../../Utils/NominalToReal'
import { retirementYears } from '../../../Utils/RetirementYears'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { assert, fGet, noCase } from '../../../Utils/Utils'
import { SimulationInfo } from '../../App/WithSimulation'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType
} from './ChartPanelType'

export type TPAWChartData = {
  label:string
  percentiles: {
    data: (x: number) => number
    percentile: number
    isHighlighted: boolean
  }[]
  min: {x: number; y: number}
  max: {x: number; y: number}
  age: {start: number; retirement: number; end: number}
  modelAgeEnd: number
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
    label:`${curr.label} scaled.`,
    percentiles: curr.percentiles.map(({data, percentile, isHighlighted}) => ({
      data: x => scaleY(data(x)),
      percentile,
      isHighlighted,
    })),
    min: {x: curr.min.x, y: scaleY(curr.min.y)},
    max: {x: curr.max.x, y: scaleY(curr.max.y)},
    age: curr.age,
    modelAgeEnd: curr.modelAgeEnd,
    isAgeInGroup: curr.isAgeInGroup,
  }
}

export const tpawChartData = (
  type: ChartPanelType,
  tpawResult: SimulationInfo['tpawResult'],
  highlightPercentiles: SimulationInfo['highlightPercentiles']
): TPAWChartData => {
  const {params} = tpawResult.args
  const {legacy, spendingCeiling, withdrawals} = params
  const hasLegacy = legacy.total !== 0 || spendingCeiling !== null
  const spendingYears = [
    ...params.withdrawals.fundedByBonds,
    ...params.withdrawals.fundedByRiskPortfolio,
  ].some(x => numericYear(params, x.yearRange.start) < params.age.retirement)
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
  const yearRange = numericYearRange(params, extraWithdrawal.yearRange)
  const yearRangeFromNow = {
    start: yearRange.start - params.age.start,
    end: yearRange.end - params.age.start,
  }

  const byPercentileByYearsFromNow = x.byPercentileByYearsFromNow.map(
    ({data, percentile}) => ({
      data: data.map((value, yearFromNow) => {
        if (
          yearFromNow < yearRangeFromNow.start ||
          yearFromNow > yearRangeFromNow.end
        ) {
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
  label:string,
  tpawResult: UseTPAWWorkerResult,
  dataFn: (
    tpawResult: UseTPAWWorkerResult
  ) => TPAWRunInWorkerByPercentileByYearsFromNow,
  years: 'retirementYears' | 'allYears',
  yearEndDelta: number,
  ageGroups: SimpleRange[],
  highlightPercentiles: number[]
): TPAWChartData => {
  const {args} = tpawResult
  const age = {
    start:
      years === 'retirementYears'
        ? args.params.age.retirement
        : args.params.age.start,
    end: args.params.age.end + yearEndDelta,
    retirement: args.params.age.retirement,
  }
  const modelAgeEnd = args.params.age.end
  const {byPercentileByYearsFromNow} = dataFn(tpawResult)
  const byPercentileByYears = byPercentileByYearsFromNow.map(({data}) =>
    years === 'retirementYears'
      ? {data: retirementYears(args.params, data)}
      : {data}
  )
  const percentiles = _addPercentileInfo(
    _interpolate(byPercentileByYears, age.start),
    args.percentiles,
    highlightPercentiles
  )

  const last = fGet(_.last(byPercentileByYears)).data
  const first = fGet(_.first(byPercentileByYears)).data
  const maxY = Math.max(...last)
  const minY = Math.min(...first)
  const indexXToDataX = (indexX: number) =>
    indexX +
    (years === 'retirementYears'
      ? args.params.age.retirement
      : args.params.age.start)
  const max = {x: indexXToDataX(last.indexOf(maxY)), y: maxY}
  const min = {x: indexXToDataX(first.indexOf(minY)), y: minY}

  const isAgeInGroup = (age: number) =>
    ageGroups.some(x => x.start <= age && x.end >= age)
  return {label, age, percentiles, min, max, modelAgeEnd, isAgeInGroup}
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

const _interpolate = (ys: {data: number[]}[], xStart: number) => {
  const beforeSlope = _avgSlope(ys, 0)
  const afterSlope = _avgSlope(ys, ys[0].data.length - 1)
  const xEnd = xStart + ys[0].data.length - 1
  return ys.map(({data}) => {
    const extrapolateBefore = linearFnFromPointAndSlope(
      xStart,
      data[0],
      beforeSlope
    )
    const extrapolateAfter = linearFnFromPointAndSlope(
      xEnd,
      fGet(_.last(data)),
      afterSlope
    )
    const interpolated = data
      .slice(0, -1)
      .map((v, i) =>
        linearFnFomPoints(i + xStart, v, i + xStart + 1, data[i + 1])
      )
    const dataFn = (a: number) => {
      if (a <= xStart) return extrapolateBefore(a)
      if (a >= xEnd) return extrapolateAfter(a)
      const section = Math.floor(a) - xStart
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
