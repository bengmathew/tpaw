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
import { ParamsExtended } from '../ExtentParams'
import { sampledAnnualReturnStatsMap } from './GetAnnualToMonthlyRateConvertionCorrection'

export const planParamsProcessRisk = (
  paramsExt: ParamsExtended,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { params, numMonths, numRetirementMonths, currentMonth } = paramsExt

  const tpawGlidePath = _tpawGlidePath(paramsExt, expectedAnnualReturns)

  const tpaw = {
    allocation: normalizeGlidePath(
      {
        start: {
          month: currentMonth,
          stocks: tpawGlidePath.now.stockAllocation,
        },
        intermediate: [],
        end: { stocks: tpawGlidePath.atMaxAge.stockAllocation },
      },
      paramsExt,
    ),
    allocationForLegacy: {
      stocks: _mertonsFormula(
        expectedAnnualReturns,
        params.plan.risk.tpaw.riskTolerance.at20 +
          params.plan.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
        0, // Does not matter.
        0, // Does not matter.
        params.plan.advanced.sampling,
        params.plan.advanced.monteCarloSampling.blockSize,
      ).stockAllocation,
    },
  }

  const tpawAndSPAW = {
    monthlySpendingTilt: _.times(numMonths, (x) =>
      params.plan.advanced.strategy === 'SPAW'
        ? annualToMonthlyReturnRate(params.plan.risk.spaw.annualSpendingTilt)
        : linearFnFomPoints(
            0,
            tpawGlidePath.now.monthlySpendingTilt,
            numMonths - 1,
            tpawGlidePath.atMaxAge.monthlySpendingTilt,
          )(x),
    ),
    lmp: params.plan.risk.tpawAndSPAW.lmp,
  }

  const spawAndSWR = {
    allocation: normalizeGlidePath(
      params.plan.risk.spawAndSWR.allocation,
      paramsExt,
    ),
  }

  const swr = {
    monthlyWithdrawal: (() => {
      switch (params.plan.risk.swr.withdrawal.type) {
        case 'default':
          return {
            type: 'asPercent' as const,
            percent:
              DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(numRetirementMonths) / 12,
          }
        case 'asPercentPerYear':
          return {
            type: 'asPercent' as const,
            percent: params.plan.risk.swr.withdrawal.percentPerYear / 12,
          }
        case 'asAmountPerMonth':
          return {
            type: 'asAmount' as const,
            amount: params.plan.risk.swr.withdrawal.amountPerMonth,
          }
        default:
          noCase(params.plan.risk.swr.withdrawal)
      }
    })(),
  }

  return { tpaw, tpawAndSPAW, spawAndSWR, swr }
}

const _tpawGlidePath = (
  paramsExt: ParamsExtended,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { params, pickPerson, longerLivedPerson, getCurrentAgeOfPerson } =
    paramsExt
  const riskTolerance = (() => {
    const person = pickPerson(longerLivedPerson)
    const atAge = (x: number) =>
      Math.max(
        RISK_TOLERANCE_VALUES.DATA[0],
        linearFnFomPoints(
          20 * 12,
          params.plan.risk.tpaw.riskTolerance.at20,
          person.ages.maxAge.inMonths,
          params.plan.risk.tpaw.riskTolerance.at20 +
            params.plan.risk.tpaw.riskTolerance.deltaAtMaxAge,
        )(x),
      )
    return {
      now: atAge(getCurrentAgeOfPerson(longerLivedPerson).inMonths),
      atMaxAge: atAge(person.ages.maxAge.inMonths),
    }
  })()

  const deltaOverPlanningYears = riskTolerance.atMaxAge - riskTolerance.now

  const _currMertonFormula = (riskTolerance: number) =>
    _mertonsFormula(
      expectedAnnualReturns,
      riskTolerance,
      params.plan.risk.tpaw.timePreference,
      params.plan.risk.tpaw.additionalAnnualSpendingTilt,
      params.plan.advanced.sampling,
      params.plan.advanced.monteCarloSampling.blockSize,
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

export const normalizeGlidePath = (
  { start, intermediate, end }: GlidePath,
  paramsExt: ParamsExtended,
) => {
  const { asMFN, numMonths } = paramsExt
  // Keeps only the first  of duplicates and prioritizes "start" and "end", over
  // intermediate if they refer to same month.
  const stage1 = [
    //  Keeps the *first* of duplicate element, per documentation.
    ..._.uniqBy(
      // Stable sort, per documentation.
      _.sortBy(
        [
          {
            month: asMFN(start.month),
            stocks: start.stocks,
          },
          { month: numMonths - 1, stocks: end.stocks },
          ...intermediate.map(({ month, stocks }) => ({
            month: asMFN(month),
            stocks,
          })),
        ],
        (x) => x.month,
      ),
      (x) => x.month,
    ),
  ]
  assert(stage1[0].month <= 0)

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
    _.range(curr.month, next.month + 1).forEach((x) => {
      if (x >= 0 && x < numMonths) result[x] = fn(x)
    })
  })
  assert(result.length === numMonths)
  return result
}
