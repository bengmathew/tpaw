import { PLAN_PARAMS_CONSTANTS, assert, block } from '@tpaw/common'
import { BaseAndLogStats } from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { noCase } from '../../Utils/Utils'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { WASM } from './GetWASM'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'

export function runSimulationInWASM(
  currentPortfolioBalanceAmount: number,
  planParamsNorm: PlanParamsNormalized,
  planParamsProcessed: PlanParamsProcessed,
  runsSpec: SimpleRange,
  randomSeed: number,
  wasm: WASM,
  opts: {
    forFirstMonth: boolean
    test?: { truth: number[]; indexIntoHistoricalReturns: number[] }
  } = { forFirstMonth: false },
): RunSimulationInWASMResult {
  let start0 = performance.now()
  const { ages } = planParamsNorm

  const numMonthsToSimulate = opts.forFirstMonth
    ? 1
    : ages.simulationMonths.numMonths

  let start = performance.now()
  let runs = wasm.run(
    planParamsNorm.advanced.strategy,
    runsSpec.start,
    runsSpec.end,
    ages.simulationMonths.numMonths,
    numMonthsToSimulate,
    ages.simulationMonths.withdrawalStartMonth.asMFN,
    planParamsProcessed.expectedReturnsForPlanning.monthlyNonLogForSimulation
      .stocks,
    planParamsProcessed.expectedReturnsForPlanning.monthlyNonLogForSimulation
      .bonds,
    planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.logSeries,
    planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.logSeries,
    currentPortfolioBalanceAmount,
    Float64Array.from(planParamsProcessed.risk.tpaw.allocation),
    Float64Array.from(planParamsProcessed.risk.spawAndSWR.allocation),
    planParamsProcessed.risk.tpaw.allocationForLegacy.stocks,
    planParamsProcessed.risk.swr.monthlyWithdrawal.type,
    planParamsProcessed.risk.swr.monthlyWithdrawal.type === 'asAmount'
      ? planParamsProcessed.risk.swr.monthlyWithdrawal.amount
      : planParamsProcessed.risk.swr.monthlyWithdrawal.type === 'asPercent'
        ? planParamsProcessed.risk.swr.monthlyWithdrawal.percent
        : noCase(planParamsProcessed.risk.swr.monthlyWithdrawal),
    planParamsProcessed.byMonth.risk.tpawAndSPAW.lmp,
    planParamsProcessed.byMonth.wealth.total,
    planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending.essential
      .total,
    planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending
      .discretionary.total,
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.target,
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.external,
    Float64Array.from(planParamsProcessed.risk.tpawAndSPAW.monthlySpendingTilt),
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW
      .monthlySpendingCeiling ?? undefined,
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW
      .monthlySpendingFloor ?? undefined,
    planParamsNorm.advanced.sampling.type === 'monteCarlo'
      ? true
      : planParamsNorm.advanced.sampling.type === 'historical'
        ? false
        : noCase(planParamsNorm.advanced.sampling.type),
    planParamsNorm.advanced.sampling.forMonteCarlo.blockSize.inMonths,
    planParamsNorm.advanced.sampling.forMonteCarlo.staggerRunStarts,
    PLAN_PARAMS_CONSTANTS.maxAgeInMonths,
    BigInt(randomSeed),
    opts.test?.truth ? Float64Array.from(opts.test.truth) : undefined,
    opts.test?.indexIntoHistoricalReturns
      ? Uint32Array.from(opts.test.indexIntoHistoricalReturns)
      : undefined,
  )
  const perfRuns = performance.now() - start

  const numRuns = runsSpec.end - runsSpec.start
  const monthIndexes = _.range(0, numMonthsToSimulate)
  const splitArray = (x: Float64Array) => {
    const copy = x.slice()
    return monthIndexes.map((month) =>
      copy.subarray(month * numRuns, (month + 1) * numRuns),
    )
  }

  start = performance.now()

  const result: Omit<RunSimulationInWASMResult, 'perf'> = {
    byMonthsFromNowByRun: {
      savingsPortfolio: {
        start: {
          balance: splitArray(runs.by_mfn_by_run_balance_start()),
        },
        withdrawals: {
          essential: splitArray(runs.by_mfn_by_run_withdrawals_essential()),
          discretionary: splitArray(
            runs.by_mfn_by_run_withdrawals_discretionary(),
          ),
          regular: splitArray(runs.by_mfn_by_run_withdrawals_regular()),
          total: splitArray(runs.by_mfn_by_run_withdrawals_total()),
          fromSavingsPortfolioRate: splitArray(
            runs.by_mfn_by_run_withdrawals_from_savings_portfolio_rate(),
          ),
        },
        afterWithdrawals: {
          allocation: {
            stocks: splitArray(
              runs.by_mfn_by_run_after_withdrawals_allocation_stocks_savings(),
            ),
          },
        },
      },
      totalPortfolio: {
        afterWithdrawals: {
          allocation: {
            stocks: splitArray(
              runs.by_mfn_by_run_after_withdrawals_allocation_stocks_total(),
            ),
          },
        },
      },
    },
    byRun: {
      numInsufficientFundMonths: runs
        .by_run_num_insufficient_fund_months()
        .slice(),
      endingBalanceOfSavingsPortfolio: runs.by_run_ending_balance().slice(),
    },
    annualStatsForSampledReturns: block(() => {
      const processStatsForWindowSize = (
        stats: BaseAndLogStats | null | undefined,
      ) => {
        if (stats) return { ofBase: stats.base, ofLog: stats.log }
        else {
          assert(opts.forFirstMonth)
          let zero = { mean: 0, variance: 0, standardDeviation: 0, n: 0 }
          return { ofBase: zero, ofLog: zero }
        }
      }
      return {
        stocks: processStatsForWindowSize(
          runs.annual_stats_for_sampled_stock_returns(),
        ),
        bonds: processStatsForWindowSize(
          runs.annual_stats_for_sampled_bond_returns(),
        ),
      }
    }),
  }
  runs.free()
  const perfPost = performance.now() - start
  const perfTotal = performance.now() - start0
  const perfRest = perfTotal - perfRuns - perfPost
  return {
    ...result,
    perf: [
      ['runs', perfRuns],
      ['post', perfPost],
      ['rest', perfRest],
      ['total', perfTotal],
    ],
  }
}
