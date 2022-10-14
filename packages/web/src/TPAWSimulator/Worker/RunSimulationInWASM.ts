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
    params.currentPortfolioBalance,
    Float64Array.from(params.risk.tpaw.allocation),
    Float64Array.from(params.risk.spawAndSWR.allocation),
    params.risk.tpaw.allocationForLegacy.stocks,
    params.risk.swr.withdrawal.type,
    params.risk.swr.withdrawal.type === 'asAmount'
      ? params.risk.swr.withdrawal.amount
      : params.risk.swr.withdrawal.type === 'asPercent'
      ? params.risk.swr.withdrawal.percent
      : noCase(params.risk.swr.withdrawal),
    Float64Array.from(
      params.byYear.map(x => x.tpawAndSPAW.risk.lmp)
    ),
    Float64Array.from(params.byYear.map(x => x.futureSavingsAndRetirementIncome)),
    Float64Array.from(params.byYear.map(x => x.extraSpending.essential)),
    Float64Array.from(params.byYear.map(x => x.extraSpending.discretionary)),
    params.legacy.tpawAndSPAW.target,
    params.legacy.tpawAndSPAW.external,
    params.risk.tpawAndSPAW.spendingTilt,
    params.risk.tpawAndSPAW.spendingCeiling ?? undefined,
    params.risk.tpawAndSPAW.spendingFloor ?? undefined,
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
              runs.by_yfn_by_run_after_withdrawals_allocation_stocks_savings()
            ),
          },
        },
      },
      totalPortfolio: {
        afterWithdrawals: {
          allocation: {
            stocks: splitArray(
              runs.by_yfn_by_run_after_withdrawals_allocation_stocks_total()
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
