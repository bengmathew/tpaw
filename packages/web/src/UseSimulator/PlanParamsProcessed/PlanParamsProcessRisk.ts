import {
  annualToMonthlyReturnRate,
  assert,
  block,
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  GlidePath,
  letIn,
  linearFnFomPoints,
  monthlyToAnnualReturnRate,
  noCase,
  RISK_TOLERANCE_VALUES,
} from '@tpaw/common'
import _ from 'lodash'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { PlanParamsExtended } from '../ExtentPlanParams'

export const planParamsProcessRisk = (
  planParamsExt: PlanParamsExtended,
  expectedReturnsForPlanning: { annual: { stocks: number; bonds: number } },
  estimatedSampledStatsOfStocks: { ofLog: { variance: number } },
) => {
  const { planParams, numMonths, numRetirementMonths } = planParamsExt

  const tpawGlidePath = _getTPAWGlidePath(
    planParamsExt,
    expectedReturnsForPlanning,
    estimatedSampledStatsOfStocks,
  )
  // _logTPAWGlidePath(tpawGlidePath)

  const tpaw = {
    fullGlidePath: tpawGlidePath,
    allocation: tpawGlidePath.map(
      (x) => x.clamped?.stockAllocation ?? x.unclamped.stockAllocation,
    ),
    allocationForLegacy: {
      stocks: _.clamp(
        _pureMertonsFormula(
          expectedReturnsForPlanning,
          estimatedSampledStatsOfStocks,
          letIn(
            planParams.risk.tpaw.riskTolerance.at20 +
              planParams.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
            (rra) =>
              RISK_TOLERANCE_VALUES.riskToleranceToRRA.withInfinityAtZero(rra),
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
      planParams.advanced.strategy === 'SPAW'
        ? _.times(numMonths, (x) =>
            annualToMonthlyReturnRate(planParams.risk.spaw.annualSpendingTilt),
          )
        : tpawGlidePath.map(
            (x) =>
              x.clamped?.monthlySpendingTilt ?? x.unclamped.monthlySpendingTilt,
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

const _getTPAWGlidePath = (
  planParamsExt: PlanParamsExtended,
  expectedReturnsForPlanning: { annual: { stocks: number; bonds: number } },
  estimatedSampledStatsOfStocks: { ofLog: { variance: number } },
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
  const { planParams, getRiskToleranceFromMFN, numMonths } = planParamsExt

  const _currMertonFormula = (rra: number) =>
    _pureMertonsFormula(
      expectedReturnsForPlanning,
      estimatedSampledStatsOfStocks,
      rra,
      planParams.risk.tpaw.timePreference,
      planParams.risk.tpaw.additionalAnnualSpendingTilt,
    )

  const mertonsFormulaByMFN = (mfn: number, minRRA: number) => {
    const unclamped = block(() => {
      const riskTolerance = getRiskToleranceFromMFN(mfn)
      const rra =
        RISK_TOLERANCE_VALUES.riskToleranceToRRA.withInfinityAtZero(
          riskTolerance,
        )
      return { riskTolerance, rra, ..._currMertonFormula(rra) }
    })
    const nominallyClampStockAllocation = (stockAllocation: number) => {
      const result = _.clamp(stockAllocation, 0, 1)
      // stockAllocation should already be in the range for [0, 1], but for
      // floating point imprecision.
      assert(Math.abs(result - stockAllocation) < 0.0001)
      return result
    }
    if (minRRA > unclamped.rra) {
      const clamped = block(() => {
        const rra = Math.max(minRRA, unclamped.rra)
        const { stockAllocation, monthlySpendingTilt } = _currMertonFormula(rra)
        return {
          rra,
          riskToleranceByInversion:
            RISK_TOLERANCE_VALUES.riskToleranceToRRA.withoutInfinityAtZero.inverse(
              rra,
            ),
          stockAllocation,
          monthlySpendingTilt,
        }
      })
      return { minRRA, unclamped, clamped }
    } else {
      unclamped.stockAllocation = nominallyClampStockAllocation(
        unclamped.stockAllocation,
      )
      return { minRRA, unclamped, clamped: null }
    }
  }

  const equityPremium =
    expectedReturnsForPlanning.annual.stocks -
    expectedReturnsForPlanning.annual.bonds

  if (equityPremium < 0) {
    // Mertons formula yields a stock allocation of -Infinity and 0 for rra of 0
    // and Infinity respectively. But we don't really want to handle this case
    // using Merton's formula, because negative stock allocation means leverage,
    // which we don't allow. We want instead to completely ignore stocks. We do
    // this by bringing equity premium to 0.
    return _getTPAWGlidePath(
      planParamsExt,
      {
        annual: {
          stocks: expectedReturnsForPlanning.annual.bonds, // Note, this is bonds.
          bonds: expectedReturnsForPlanning.annual.bonds,
        },
      },
      estimatedSampledStatsOfStocks,
    )
  } else if (equityPremium === 0) {
    // Mertons formula yields a stock allocation of 0 for the full rra range
    // from 0 to Infinity. This is always in range, so no need to clamp.
    return _.times(numMonths, (mfn) => mertonsFormulaByMFN(mfn, 0))
  } else {
    // Mertons formula yields a stock allocation of Infinity and 0 for rra of 0
    // and Infinity respectively. We can go through mertons formula, but we clamp
    // rra to the rra that results in a stock allocation of 1.
    const minRRA = _pureMertonsFormula.inverseFromStockAllocation(
      expectedReturnsForPlanning,
      estimatedSampledStatsOfStocks,
      1,
    ).rra

    return _.times(numMonths, (mfn) => mertonsFormulaByMFN(mfn, minRRA))
  }
}

// Note: This is pure. It does not clamp stock allocation.
export const _pureMertonsFormula = (
  expectedReturnsForPlanning: { annual: { stocks: number; bonds: number } },
  estimatedSampledStatsOfStocks: { ofLog: { variance: number } },
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
  const r = expectedReturnsForPlanning.annual.bonds
  const mu = expectedReturnsForPlanning.annual.stocks
  const sigmaPow2 = estimatedSampledStatsOfStocks.ofLog.variance
  const stockAllocation = (mu - r) / (sigmaPow2 * gamma)

  const rho = timePreference
  const nu =
    (rho - (1 - gamma) * (Math.pow(mu - r, 2) / (2 * sigmaPow2 * gamma) + r)) /
    gamma
  const rOfPortfolio = mu * stockAllocation + r * (1 - stockAllocation)
  const annualSpendingTilt = rOfPortfolio - nu

  return result({ annualSpendingTilt, stockAllocation })
}

_pureMertonsFormula.inverseFromStockAllocation = (
  expectedReturnsForPlanning: { annual: { stocks: number; bonds: number } },
  estimatedSampledStatsOfStocks: { ofLog: { variance: number } },
  stockAllocation: number,
) => {
  const r = expectedReturnsForPlanning.annual.bonds
  const mu = expectedReturnsForPlanning.annual.stocks
  const sigmaPow2 = estimatedSampledStatsOfStocks.ofLog.variance
  const gamma = (mu - r) / (sigmaPow2 * stockAllocation)
  // Not returning riskTolerance because it may be < 0.
  return { rra: gamma }
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
