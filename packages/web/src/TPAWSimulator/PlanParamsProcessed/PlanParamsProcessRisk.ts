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
import { PlanParamsExtended } from '../ExtentPlanParams'
import { sampledAnnualReturnStatsMap } from './GetAnnualToMonthlyRateConvertionCorrection'

export const planParamsProcessRisk = (
  planParamsExt: PlanParamsExtended,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { planParams, numMonths, numRetirementMonths, currentMonth } =
    planParamsExt

  const tpawGlidePath = _tpawGlidePath(planParamsExt, expectedAnnualReturns)

  const tpaw = {
    allocation: normalizeGlidePath(
      {
        start: {
          month: currentMonth,
          stocks: tpawGlidePath.now.stockAllocation,
        },
        intermediate: {},
        end: { stocks: tpawGlidePath.atMaxAge.stockAllocation },
      },
      planParamsExt,
    ),
    allocationForLegacy: {
      stocks: _mertonsFormula(
        expectedAnnualReturns,
        planParams.risk.tpaw.riskTolerance.at20 +
          planParams.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
        0, // Does not matter.
        0, // Does not matter.
        planParams.advanced.sampling,
        planParams.advanced.monteCarloSampling.blockSize,
      ).stockAllocation,
    },
  }

  const tpawAndSPAW = {
    monthlySpendingTilt: _.times(numMonths, (x) =>
      planParams.advanced.strategy === 'SPAW'
        ? annualToMonthlyReturnRate(planParams.risk.spaw.annualSpendingTilt)
        : linearFnFomPoints(
            0,
            tpawGlidePath.now.monthlySpendingTilt,
            numMonths - 1,
            tpawGlidePath.atMaxAge.monthlySpendingTilt,
          )(x),
    ),
    lmp: planParams.risk.tpawAndSPAW.lmp,
  }

  const spawAndSWR = {
    allocation: normalizeGlidePath(
      planParams.risk.spawAndSWR.allocation,
      planParamsExt,
    ),
  }

  const swr = {
    monthlyWithdrawal: (() => {
      switch (planParams.risk.swr.withdrawal.type) {
        case 'default':
          return {
            type: 'asPercent' as const,
            percent:
              DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(numRetirementMonths) / 12,
          }
        case 'asPercentPerYear':
          return {
            type: 'asPercent' as const,
            percent: planParams.risk.swr.withdrawal.percentPerYear / 12,
          }
        case 'asAmountPerMonth':
          return {
            type: 'asAmount' as const,
            amount: planParams.risk.swr.withdrawal.amountPerMonth,
          }
        default:
          noCase(planParams.risk.swr.withdrawal)
      }
    })(),
  }

  return { tpaw, tpawAndSPAW, spawAndSWR, swr }
}

const _tpawGlidePath = (
  planParamsExt: PlanParamsExtended,
  expectedAnnualReturns: { stocks: number; bonds: number },
) => {
  const { planParams, pickPerson, longerLivedPerson, getCurrentAgeOfPerson } =
    planParamsExt
  const riskTolerance = (() => {
    const person = pickPerson(longerLivedPerson)
    const atAge = (x: number) =>
      Math.max(
        RISK_TOLERANCE_VALUES.DATA[0],
        linearFnFomPoints(
          20 * 12,
          planParams.risk.tpaw.riskTolerance.at20,
          person.ages.maxAge.inMonths,
          planParams.risk.tpaw.riskTolerance.at20 +
            planParams.risk.tpaw.riskTolerance.deltaAtMaxAge,
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
      planParams.risk.tpaw.timePreference,
      planParams.risk.tpaw.additionalAnnualSpendingTilt,
      planParams.advanced.sampling,
      planParams.advanced.monteCarloSampling.blockSize,
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
  planParamsExt: PlanParamsExtended,
) => {
  const { asMFN, numMonths } = planParamsExt
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
          ..._.values(intermediate)
            .sort((a, b) => a.indexToSortByAdded - b.indexToSortByAdded)
            .map(({ month, stocks }) => ({
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

  let result = [] as number[]
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
