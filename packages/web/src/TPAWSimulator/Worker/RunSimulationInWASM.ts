import { MAX_AGE_IN_MONTHS, block } from '@tpaw/common'
import { Stats, StatsForWindowSize } from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { noCase } from '../../Utils/Utils'
import { extendPlanParams } from '../ExtentPlanParams'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { WASM } from './GetWASM'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'

export function runSimulationInWASM(
  params: PlanParamsProcessed,
  runsSpec: SimpleRange,
  wasm: WASM,
  opts: {
    forFirstMonth: boolean
    test?: { truth: number[]; indexIntoHistoricalReturns: number[] }
  } = { forFirstMonth: false },
): RunSimulationInWASMResult {
  let start0 = performance.now()
  const { numMonths, asMFN, withdrawalStartMonth } = extendPlanParams(
    params.original,
    params.currentTime.epoch,
    params.currentTime.zoneName,
  )

  const numMonthsToSimulate = opts.forFirstMonth ? 1 : numMonths

  let start = performance.now()
  let runs = wasm.run(
    params.strategy,
    runsSpec.start,
    runsSpec.end,
    numMonths,
    numMonthsToSimulate,
    asMFN(withdrawalStartMonth),
    params.expectedReturnsForPlanning.monthly.stocks,
    params.expectedReturnsForPlanning.monthly.bonds,
    Float64Array.from(params.historicalReturnsAdjusted.monthly.stocks),
    Float64Array.from(params.historicalReturnsAdjusted.monthly.bonds),
    params.estimatedCurrentPortfolioBalance,
    Float64Array.from(params.risk.tpaw.allocation),
    Float64Array.from(params.risk.spawAndSWR.allocation),
    params.risk.tpaw.allocationForLegacy.stocks,
    params.risk.swr.monthlyWithdrawal.type,
    params.risk.swr.monthlyWithdrawal.type === 'asAmount'
      ? params.risk.swr.monthlyWithdrawal.amount
      : params.risk.swr.monthlyWithdrawal.type === 'asPercent'
      ? params.risk.swr.monthlyWithdrawal.percent
      : noCase(params.risk.swr.monthlyWithdrawal),
    params.byMonth.risk.tpawAndSPAW.lmp,
    params.byMonth.wealth.total,
    params.byMonth.adjustmentsToSpending.extraSpending.essential.total,
    params.byMonth.adjustmentsToSpending.extraSpending.discretionary.total,
    params.adjustmentsToSpending.tpawAndSPAW.legacy.target,
    params.adjustmentsToSpending.tpawAndSPAW.legacy.external,
    Float64Array.from(params.risk.tpawAndSPAW.monthlySpendingTilt),
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling ??
      undefined,
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor ?? undefined,
    params.sampling.type === 'monteCarlo'
      ? true
      : params.sampling.type === 'historical'
      ? false
      : noCase(params.sampling.type),
    params.samplingBlockSizeForMonteCarlo,
    MAX_AGE_IN_MONTHS,
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
      const processStats = (stats: Stats) => ({
        mean: stats.mean,
        variance: stats.variance,
        standardDeviation: stats.standard_deviation,
        n: stats.n,
      })
      const processStatsForWindowSize = (stats: StatsForWindowSize) => ({
        n: stats.n,
        ofBase: processStats(stats.of_base),
        ofLog: processStats(stats.of_log),
      })
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

// const getAnnualMeanFromMonthlyReturns = (monthlyReturns: Float64Array) =>
//   _.mean(
//     _.range(0, monthlyReturns.length - 12)
//       .map((i) => monthlyReturns.slice(i, i + 12))
//       .map(monthArrToYear),
//   )

// const monthArrToYear = (year: Float64Array) =>
//   year.map((x) => 1 + x).reduce(_.multiply, 1) - 1
