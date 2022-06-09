import _ from 'lodash'
import {getNumYears, getWithdrawalStartAsYFN} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'

export async function runSimulationInWASM(
  params: TPAWParamsProcessed,
  numRuns: number,
  test?: {truth: number[]; indexIntoHistoricalReturns: number[]}
) {
  let start0 = performance.now()
  const numYears = getNumYears(params.original)
  const wasm = await import('@tpaw/simulator')

  let start = performance.now()
  let runs = wasm.run(
    params.strategy,
    numRuns,
    numYears,
    getWithdrawalStartAsYFN(params.original),
    params.returns.expected.stocks,
    params.returns.expected.bonds,
    Float64Array.from(params.returns.historicalAdjusted.map(x => x.stocks)),
    Float64Array.from(params.returns.historicalAdjusted.map(x => x.bonds)),
    params.savingsAtStartOfStartYear,
    params.targetAllocation.regularPortfolio.forTPAW.stocks,
    Float64Array.from(params.targetAllocation.regularPortfolio.forSPAW),
    params.targetAllocation.legacyPortfolio.stocks,
    params.withdrawals.lmp,
    Float64Array.from(params.byYear.map(x => x.savings)),
    Float64Array.from(params.byYear.map(x => x.withdrawals.essential)),
    Float64Array.from(params.byYear.map(x => x.withdrawals.discretionary)),
    params.legacy.target,
    params.legacy.external,
    params.scheduledWithdrawalGrowthRate,
    params.spendingCeiling ?? undefined,
    params.spendingFloor ?? undefined,
    test?.truth ? Float64Array.from(test.truth) : undefined,
    test?.indexIntoHistoricalReturns
      ? Uint32Array.from(test.indexIntoHistoricalReturns)
      : undefined
  )
  const perfRuns = performance.now() - start

  const yearIndexes = _.range(0, numYears)
  const splitArray = (x: Float64Array) => {
    const copy = x.slice()
    return yearIndexes.map(year =>
      copy.subarray(year * numRuns, (year + 1) * numRuns)
    )
  }

  start = performance.now()

  const result = {
    byYearsFromNowByRun: {
      savingsPortfolio: {
        start: {
          balance: splitArray(runs.by_yfn_by_run_balance_start()),
        },
        withdrawals: {
          essential: splitArray(runs.by_yfn_by_run_withdrawals_essential()),
          discretionary: splitArray(
            runs.by_yfn_by_run_withdrawals_discretionary()
          ),
          regular: splitArray(runs.by_yfn_by_run_withdrawals_regular()),
          total: splitArray(runs.by_yfn_by_run_withdrawals_total()),
          fromSavingsPortfolioRate: splitArray(
            runs.by_yfn_by_run_withdrawals_from_savings_portfolio_rate()
          ),
        },
        afterWithdrawals: {
          allocation: {
            stocks: splitArray(
              runs.by_yfn_by_run_after_withdrawals_allocation_stocks()
            ),
          },
        },
      },
    },
    byRun: {
      endingBalanceOfSavingsPortfolio: runs.by_run_ending_balance().slice(),
    },
  }
  runs.free()

  const perfPost = performance.now() - start
  const perfTotal = performance.now() - start0
  const perfRest = perfTotal - perfRuns - perfPost
  return {
    result,
    perf: {runs: perfRuns, post: perfPost, rest: perfRest, total: perfTotal},
  }
}
