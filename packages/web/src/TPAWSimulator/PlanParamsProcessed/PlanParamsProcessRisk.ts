import {
  annualToMonthlyReturnRate,
  assert,
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  fGet,
  GlidePath,
  historicalReturns,
  linearFnFomPoints,
  noCase,
  PlanParams,
  RISK_TOLERANCE_VALUES,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsExt } from '../PlanParamsExt'
import { sampledAnnualReturnStatsMap } from './GetAnnualToMonthlyRateConvertionCorrection'

export const planParamsProcessRisk = (
  paramsExt: PlanParamsExt,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { params, numMonths, numRetirementMonths } = paramsExt

  const tpawGlidePath = _tpawGlidePath(paramsExt, expectedAnnualReturns)

  const tpaw = {
    allocation: _normalizeGlidePath(
      {
        start: { stocks: tpawGlidePath.now.stockAllocation },
        intermediate: [],
        end: { stocks: tpawGlidePath.atMaxAge.stockAllocation },
      },
      paramsExt,
    ),
    allocationForLegacy: {
      stocks: _mertonsFormula(
        expectedAnnualReturns,
        params.risk.tpaw.riskTolerance.at20 +
          params.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
        0, // Does not matter.
        0, // Does not matter.
        params.advanced.sampling,
        params.advanced.samplingBlockSizeForMonteCarlo,
      ).stockAllocation,
    },
  }

  // console.dir(monthlyReturns.expectedMonthly)
  // console.dir(
  //   params.risk.tpaw.riskTolerance.at20 +
  //     params.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
  // )
  // console.dir(tpaw.allocationForLegacy.stocks)

  const tpawAndSPAW = {
    monthlySpendingTilt: _.times(numMonths, (x) =>
      params.advanced.strategy === 'SPAW'
        ? annualToMonthlyReturnRate(params.risk.spaw.annualSpendingTilt)
        : linearFnFomPoints(
            0,
            tpawGlidePath.now.monthlySpendingTilt,
            numMonths - 1,
            tpawGlidePath.atMaxAge.monthlySpendingTilt,
          )(x),
    ),
    lmp: params.risk.tpawAndSPAW.lmp,
  }

  const spawAndSWR = {
    allocation: _normalizeGlidePath(
      params.risk.spawAndSWR.allocation,
      paramsExt,
    ),
  }

  const swr = {
    monthlyWithdrawal: (() => {
      switch (params.risk.swr.withdrawal.type) {
        case 'default':
          return {
            type: 'asPercent' as const,
            percent:
              DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(numRetirementMonths) / 12,
          }
        case 'asPercentPerYear':
          return {
            type: 'asPercent' as const,
            percent: params.risk.swr.withdrawal.percentPerYear / 12,
          }
        case 'asAmountPerMonth':
          return {
            type: 'asAmount' as const,
            amount: params.risk.swr.withdrawal.amountPerMonth,
          }
        default:
          noCase(params.risk.swr.withdrawal)
      }
    })(),
  }

  return { tpaw, tpawAndSPAW, spawAndSWR, swr }
}

const _tpawGlidePath = (
  paramsExt: PlanParamsExt,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { params, pickPerson, longerLivedPerson } = paramsExt
  const riskTolerance = (() => {
    const person = pickPerson(longerLivedPerson)
    const atAge = (x: number) =>
      Math.max(
        RISK_TOLERANCE_VALUES.DATA[0],
        linearFnFomPoints(
          20 * 12,
          params.risk.tpaw.riskTolerance.at20,
          person.ages.maxMonth,
          params.risk.tpaw.riskTolerance.at20 +
            params.risk.tpaw.riskTolerance.deltaAtMaxAge,
        )(x),
      )
    return {
      now: atAge(person.ages.currentMonth),
      atMaxAge: atAge(person.ages.maxMonth),
    }
  })()

  const deltaOverPlanningYears = riskTolerance.atMaxAge - riskTolerance.now

  const _currMertonFormula = (riskTolerance: number) =>
    _mertonsFormula(
      expectedAnnualReturns,
      riskTolerance,
      params.risk.tpaw.timePreference,
      params.risk.tpaw.additionalAnnualSpendingTilt,
      params.advanced.sampling,
      params.advanced.samplingBlockSizeForMonteCarlo,
    )

  const maxRiskTolerance =
    _.find(
      RISK_TOLERANCE_VALUES.DATA,
      (x) => _currMertonFormula(x).stockAllocation === 1,
    ) ?? fGet(_.last(RISK_TOLERANCE_VALUES.DATA))

  const riskToleranceAtMaxAge = Math.min(
    maxRiskTolerance,
    riskTolerance.atMaxAge,
  )

  return {
    now: _currMertonFormula(riskToleranceAtMaxAge - deltaOverPlanningYears),
    atMaxAge: _currMertonFormula(riskToleranceAtMaxAge),
  }
}

export const _mertonsFormula = (
  expectedAnnualReturns: { stocks: number; bonds: number },
  riskTolerance: number,
  timePreference: number,
  additionalAnnualSpendingTilt: number,
  sampling: PlanParams['advanced']['sampling'],
  samplingBlockSizeForMonteCarlo: number,
) => {
  const result = (x: {
    annualSpendingTilt: number
    stockAllocation: number
  }) => ({
    monthlySpendingTilt: annualToMonthlyReturnRate(
      x.annualSpendingTilt + additionalAnnualSpendingTilt,
    ),
    stockAllocation: x.stockAllocation,
  })
  if (riskTolerance === 0)
    return result({ annualSpendingTilt: 0, stockAllocation: 0 })

  const r = expectedAnnualReturns.bonds
  const mu = expectedAnnualReturns.stocks
  const sigmaPow2 =
    sampling === 'monteCarlo'
      ? fGet(sampledAnnualReturnStatsMap.get(samplingBlockSizeForMonteCarlo))
          .stocks.varianceOfLogAveragedOverThread
      : sampling === 'historical'
      ? historicalReturns.monthly.annualStats.stocks.ofLog.variance
      : noCase(sampling)
  const gamma = RISK_TOLERANCE_VALUES.riskToleranceToRRA(riskTolerance)

  const stockAllocation = Math.min(1, (mu - r) / (sigmaPow2 * gamma))
  const rho = timePreference

  const nu =
    (rho - (1 - gamma) * (Math.pow(mu - r, 2) / (2 * sigmaPow2 * gamma) + r)) /
    gamma

  const rOfPortfolio = mu * stockAllocation + r * (1 - stockAllocation)
  const annualSpendingTilt = rOfPortfolio - nu

  return result({ annualSpendingTilt, stockAllocation })
}

const _normalizeGlidePath = (
  glidePath: GlidePath,
  paramsExt: PlanParamsExt,
) => {
  const { asMFN, numMonths } = paramsExt
  const stage1 = [
    { month: 0, stocks: glidePath.start.stocks },
    // Keeps only the first month of duplicates, and removes now and end.
    ..._.uniqBy(
      _.sortBy(
        glidePath.intermediate.map(({ month, stocks }) => ({
          month: asMFN(month),
          stocks,
        })),
        (x) => x.month,
      ).filter((x) => x.month !== 0 && x.month !== numMonths - 1),
      (x) => x.month,
    ),
    { month: numMonths - 1, stocks: glidePath.end.stocks },
  ]

  let result = _.times(numMonths)
  _.times(stage1.length - 1, (i) => {
    const curr = stage1[i]
    const next = stage1[i + 1]
    const fn = linearFnFomPoints(
      curr.month,
      curr.stocks,
      next.month,
      next.stocks,
    )
    _.range(curr.month, next.month + 1).forEach((x) => (result[x] = fn(x)))
  })
  assert(result.length === numMonths)
  return result
}
