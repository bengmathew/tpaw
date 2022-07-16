import { RunResult } from '@tpaw/simulator'
import _ from 'lodash'
import {SimpleRange} from '../../Utils/SimpleRange'
import {noCase} from '../../Utils/Utils'
import {getNumYears, getWithdrawalStartAsYFN} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {getWASM} from './GetWASM'
import {TPAWWorkerRunSimulationResult} from './TPAWWorkerTypes'

export async function runSimulationInWASM(
  params: TPAWParamsProcessed,
  runsSpec: SimpleRange,
  test?: {truth: number[]; indexIntoHistoricalReturns: number[]}
): Promise<TPAWWorkerRunSimulationResult> {
  let start0 = performance.now()
  const numYears = getNumYears(params.original)
  const wasm = await getWASM()

  let start = performance.now()
  let runs = wasm.run(
    params.strategy,
    runsSpec.start,
    runsSpec.end,
    numYears,
    getWithdrawalStartAsYFN(params.original),
    params.returns.expected.stocks,
    params.returns.expected.bonds,
    Float64Array.from(params.returns.historicalAdjusted.map(x => x.stocks)),
    Float64Array.from(params.returns.historicalAdjusted.map(x => x.bonds)),
    params.savingsAtStartOfStartYear,
    params.targetAllocation.regularPortfolio.forTPAW.stocks,
    Float64Array.from(params.targetAllocation.regularPortfolio.forSPAWAndSWR),
    params.targetAllocation.legacyPortfolio.stocks,
    params.swrWithdrawal.type,
    params.swrWithdrawal.type === 'asAmount'
      ? params.swrWithdrawal.amount
      : params.swrWithdrawal.type === 'asPercent'
      ? params.swrWithdrawal.percent
      : noCase(params.swrWithdrawal),
    Float64Array.from(params.byYear.map(x => x.withdrawals.lmp)),
    Float64Array.from(params.byYear.map(x => x.savings)),
    Float64Array.from(params.byYear.map(x => x.withdrawals.essential)),
    Float64Array.from(params.byYear.map(x => x.withdrawals.discretionary)),
    params.legacy.target,
    params.legacy.external,
    params.scheduledWithdrawalGrowthRate,
    params.spendingCeiling ?? undefined,
    params.spendingFloor ?? undefined,
    params.sampling === 'monteCarlo'
      ? true
      : params.sampling === 'historical'
      ? false
      : noCase(params.sampling),
    test?.truth ? Float64Array.from(test.truth) : undefined,
    test?.indexIntoHistoricalReturns
      ? Uint32Array.from(test.indexIntoHistoricalReturns)
      : undefined
  )
  const perfRuns = performance.now() - start

  const numRuns = runsSpec.end - runsSpec.start
  const yearIndexes = _.range(0, numYears)
  const splitArray = (x: Float64Array) => {
    const copy = x.slice()
    return yearIndexes.map(year =>
      copy.subarray(year * numRuns, (year + 1) * numRuns)
    )
  }

  start = performance.now()

  const result: Omit<TPAWWorkerRunSimulationResult, 'perf'> = {
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
        excessWithdrawals: {
          regular: splitArray(runs.by_yfn_by_run_excess_withdrawals_regular()),
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
      numInsufficientFundYears: runs
        .by_run_num_insufficient_fund_years()
        .slice(),
      endingBalanceOfSavingsPortfolio: runs.by_run_ending_balancee().slice(),
    },
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
