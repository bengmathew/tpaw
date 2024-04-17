import {
  annualToMonthlyReturnRate,
  assert,
  block,
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  fGet,
  letIn,
  linearFnFomPoints,
  monthlyToAnnualReturnRate,
  noCase,
  PLAN_PARAMS_CONSTANTS,
} from '@tpaw/common'
import _ from 'lodash'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { NormalizedGlidePath } from '../NormalizePlanParams/NormalizeGlidePath'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { CallRust } from './CallRust'

export const planParamsProcessRisk = (
  planParamsNorm: PlanParamsNormalized,
  returnStatsForPlanning: CallRust.PlanParamsProcessed['returnsStatsForPlanning'],
) => {
  const { ages, risk } = planParamsNorm

  const rMuSigmaPow2 = _getRMuSigmaPow2(returnStatsForPlanning)

  const tpawGlidePath = _getTPAWGlidePath(planParamsNorm, rMuSigmaPow2)
  // _logTPAWGlidePath(tpawGlidePath)

  const tpaw = {
    fullGlidePath: tpawGlidePath,
    allocation: tpawGlidePath.map(
      (x) => x.clamped?.stockAllocation ?? x.unclamped.stockAllocation,
    ),
    allocationForLegacy: {
      stocks: _.clamp(
        _unclampedMertonsFormula(
          rMuSigmaPow2,
          letIn(
            risk.tpaw.riskTolerance.at20 +
              risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
            (rra) =>
              PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.riskToleranceToRRA.withInfinityAtZero(
                rra,
              ),
          ),
          0, // Does not matter.
          0, // Does not matter.
        ).stockAllocation,
        0,
        1,
      ),
    },
  }

  const tpawAndSPAW = {
    monthlySpendingTilt:
      planParamsNorm.advanced.strategy === 'SPAW'
        ? _.times(ages.simulationMonths.numMonths, (x) =>
            annualToMonthlyReturnRate(risk.spaw.annualSpendingTilt),
          )
        : tpawGlidePath.map(
            (x) =>
              x.clamped?.monthlySpendingTilt ?? x.unclamped.monthlySpendingTilt,
          ),
    lmp: risk.tpawAndSPAW.lmp,
  }

  const spawAndSWR = {
    allocation: _processGlidePath(risk.spawAndSWR.allocation, planParamsNorm),
  }

  const swr = {
    monthlyWithdrawal: (() => {
      switch (risk.swr.withdrawal.type) {
        case 'default':
          return {
            type: 'asPercent' as const,
            percent:
              DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                ages.simulationMonths.numWithdrawalMonths,
              ) / 12,
          }
        case 'asPercentPerYear':
          return {
            type: 'asPercent' as const,
            percent: risk.swr.withdrawal.percentPerYear / 12,
          }
        case 'asAmountPerMonth':
          return {
            type: 'asAmount' as const,
            amount: risk.swr.withdrawal.amountPerMonth,
          }
        default:
          noCase(risk.swr.withdrawal)
      }
    })(),
  }

  return { tpaw, tpawAndSPAW, spawAndSWR, swr }
}

const _getTPAWGlidePath = (
  planParamsNorm: PlanParamsNormalized,
  rMuSigmaPow2: _RMuSigmaPow2,
): {
  minRRA: number
  unclamped: {
    stockAllocation: number
    monthlySpendingTilt: number
    riskTolerance: number
    rra: number
  }
  clamped: {
    stockAllocation: number
    monthlySpendingTilt: number
    riskToleranceByInversion: number
    rra: number
  } | null
}[] => {
  const { ages, risk } = planParamsNorm
  const { numMonths } = ages.simulationMonths
  const getRiskToleranceFromMFN = (mfn: number) => {
    const longerLivedPerson = fGet(ages[ages.longerLivedPersonType])
    return Math.max(
      0,
      longerLivedPerson.maxAge.baseValue.inMonths > 20 * 12
        ? linearFnFomPoints(
            20 * 12,
            risk.tpaw.riskTolerance.at20,
            longerLivedPerson.maxAge.baseValue.inMonths,
            risk.tpaw.riskTolerance.at20 +
              risk.tpaw.riskTolerance.deltaAtMaxAge,
          )(mfn + longerLivedPerson.currentAgeInfo.inMonths)
        : risk.tpaw.riskTolerance.at20,
    )
  }

  const _currMertonFormula = (rra: number) =>
    _unclampedMertonsFormula(
      rMuSigmaPow2,
      rra,
      risk.tpaw.timePreference,
      risk.tpaw.additionalAnnualSpendingTilt,
    )

  const mertonsFormulaByMFN = (mfn: number, minRRA: number) => {
    const unclamped = block(() => {
      const riskTolerance = getRiskToleranceFromMFN(mfn)
      const rra =
        PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.riskToleranceToRRA.withInfinityAtZero(
          riskTolerance,
        )
      return { riskTolerance, rra, ..._currMertonFormula(rra) }
    })

    if (minRRA > unclamped.rra) {
      const clamped = block(() => {
        const rra = Math.max(minRRA, unclamped.rra)
        const { stockAllocation, monthlySpendingTilt } = _currMertonFormula(rra)
        return {
          rra,
          riskToleranceByInversion:
            PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.riskToleranceToRRA.withoutInfinityAtZero.inverse(
              rra,
            ),
          stockAllocation,
          monthlySpendingTilt,
        }
      })
      return { minRRA, unclamped, clamped }
    } else {
      unclamped.stockAllocation = block(() => {
        const nominallyClamped = _.clamp(unclamped.stockAllocation, 0, 1)
        // stockAllocation should already be in the range for [0, 1], but for
        // floating point imprecision.
        assert(Math.abs(nominallyClamped - unclamped.stockAllocation) < 0.0001)
        return nominallyClamped
      })
      return { minRRA, unclamped, clamped: null }
    }
  }

  // mu:stocks, r:bonds.
  const equityPremium = rMuSigmaPow2.mu - rMuSigmaPow2.r
  if (equityPremium < 0) {
    // Mertons formula yields a stock allocation of -Infinity and 0 for rra of 0
    // and Infinity respectively. But we don't really want to handle this case
    // using Merton's formula, because negative stock allocation means leverage,
    // which we don't allow. We want instead to completely ignore stocks. We do
    // this by bringing equity premium to 0.
    return _getTPAWGlidePath(planParamsNorm, {
      // bonds
      r: rMuSigmaPow2.r,
      // stocks
      mu: rMuSigmaPow2.r, // Note this is r, not mu
      sigmaPow2: rMuSigmaPow2.sigmaPow2,
    })
  } else if (equityPremium === 0) {
    // Mertons formula yields a stock allocation of 0 for the full rra range
    // from 0 to Infinity. This is always in range, so no need to clamp.
    return _.times(numMonths, (mfn) => mertonsFormulaByMFN(mfn, 0))
  } else {
    // Mertons formula yields a stock allocation of Infinity and 0 for rra of 0
    // and Infinity respectively. We can go through mertons formula, but we clamp
    // rra to the rra that results in a stock allocation of 1.
    const minRRA = _unclampedMertonsFormula.inverseFromStockAllocation(
      rMuSigmaPow2,
      1,
    ).rra

    return _.times(numMonths, (mfn) => mertonsFormulaByMFN(mfn, minRRA))
  }
}

type _RMuSigmaPow2 = {
  r: number
  mu: number
  sigmaPow2: number
}

const _getRMuSigmaPow2 = ({
  stocks,
  bonds,
}: CallRust.PlanParamsProcessed['returnsStatsForPlanning']) => ({
  r: bonds.empiricalAnnualNonLogExpectedReturnInfo.value,
  mu: stocks.empiricalAnnualNonLogExpectedReturnInfo.value,
  sigmaPow2: stocks.empiricalAnnualLogVariance,
})

// Does not clamp stock allocation.
const _unclampedMertonsFormula = (
  { r, mu, sigmaPow2 }: _RMuSigmaPow2,
  // Take in rra not riskTolerance because we need to support rra that are not
  // in range of the defined riskTolerances because we might get this rra from
  // running the inverse().
  rra: number,
  timePreference: number,
  additionalAnnualSpendingTilt: number,
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
  if (rra === Infinity)
    return result({ annualSpendingTilt: 0, stockAllocation: 0 })

  const gamma = rra

  const stockAllocation = (mu - r) / (sigmaPow2 * gamma)

  const rho = timePreference
  const nu =
    (rho - (1 - gamma) * (Math.pow(mu - r, 2) / (2 * sigmaPow2 * gamma) + r)) /
    gamma
  const rOfPortfolio = mu * stockAllocation + r * (1 - stockAllocation)
  const annualSpendingTilt = rOfPortfolio - nu

  return result({ annualSpendingTilt, stockAllocation })
}

_unclampedMertonsFormula.inverseFromStockAllocation = (
  { r, mu, sigmaPow2 }: _RMuSigmaPow2,
  stockAllocation: number,
) => {
  const gamma = (mu - r) / (sigmaPow2 * stockAllocation)
  // Not returning riskTolerance because it may be < 0.
  return { rra: gamma }
}

const _processGlidePath = (
  { now, intermediate, end }: NormalizedGlidePath,
  planParamsNorm: PlanParamsNormalized,
) => {
  const { ages } = planParamsNorm
  const simplifiedPath = [
    { mfn: 0, stocks: now.stocks },
    ...intermediate
      .filter((x) => !x.ignore)
      .map((x) => ({ mfn: x.month.asMFN, stocks: x.stocks })),
    { mfn: ages.simulationMonths.lastMonthAsMFN, stocks: end.stocks },
  ]

  let result = [] as number[]
  simplifiedPath.forEach((x, i) => {
    if (i === 0) return
    const from = simplifiedPath[i - 1]
    const to = x
    const fn = linearFnFomPoints(from.mfn, from.stocks, to.mfn, to.stocks)
    result.push(..._.range(from.mfn, to.mfn).map(fn))
  })
  result.push(end.stocks)
  assert(result.length === ages.simulationMonths.numMonths)
  return result
}

const _logTPAWGlidePath = (
  tpawGlidePath: ReturnType<typeof _getTPAWGlidePath>,
) => {
  console.dir('------------------------------')
  console.dir('------------------------------')
  console.dir(
    tpawGlidePath
      .filter((x, i) => i % 12 === 0)
      .map((x, i) => {
        const info = (
          y: Exclude<typeof x.clamped | typeof x.unclamped, null>,
        ) =>
          `rra: ${y.rra.toFixed(2).padEnd(10)} tolerance: ${('riskTolerance' in
          y
            ? y.riskTolerance
            : y.riskToleranceByInversion
          )
            .toFixed(2)
            .padEnd(10)} stocks: ${formatPercentage(2)(
            y.stockAllocation,
          ).padEnd(10)}  tilt: ${formatPercentage(2)(
            monthlyToAnnualReturnRate(y.monthlySpendingTilt),
          ).padEnd(10)}`

        const content = x.clamped
          ? `${info(x.clamped)}   CLAMPED FROM: ${info(x.unclamped)}`
          : info(x.unclamped)

        return `yfn: ${i.toFixed(0).padEnd(5)} minRRA: ${x.minRRA
          .toFixed(2)
          .padEnd(10)} ${content}`
      })
      .join('\n'),
  )
}
