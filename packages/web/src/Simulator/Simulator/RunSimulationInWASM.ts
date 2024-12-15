import { PLAN_PARAMS_CONSTANTS, assert, block } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { noCase } from '../../Utils/Utils'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { WASM } from './GetWASM'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'

export function runSimulationInWASM(
  currentPortfolioBalanceAmount: number,
  planParamsRust: Rust.PlanParamsRust,
  marketData: Rust.DataForMarketBasedPlanParamValues,
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
    planParamsRust,
    marketData,
    runsSpec.start,
    runsSpec.end,
    numMonthsToSimulate,
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
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.target,
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.external,
    Float64Array.from(planParamsProcessed.risk.tpawAndSPAW.monthlySpendingTilt),
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW
      .monthlySpendingCeiling ?? undefined,
    planParamsProcessed.adjustmentsToSpending.tpawAndSPAW
      .monthlySpendingFloor ?? undefined,
    PLAN_PARAMS_CONSTANTS.people.ages.person.maxAge,
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
        stats: Rust.BaseAndLogStats | null | undefined,
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
    info: runs.info,
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
