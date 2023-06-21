import { annualToMonthlyReturnRate, MAX_AGE_IN_MONTHS } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { SimpleRange } from '../../Utils/SimpleRange'
import { noCase } from '../../Utils/Utils'
import { extendParams } from '../ExtentParams'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { WASM } from './GetWASM'
import { TPAWWorkerRunSimulationResult } from './TPAWWorkerAPI'

export function runSimulationInWASM(
  params: PlanParamsProcessed,
  runsSpec: SimpleRange,
  wasm: WASM,
  test?: { truth: number[]; indexIntoHistoricalReturns: number[] },
): TPAWWorkerRunSimulationResult {
  let start0 = performance.now()
  const { numMonths, asMFN, withdrawalStartMonth } = extendParams(
    params.original,
    DateTime.fromMillis(params.currentTime.epoch, {
      zone: params.currentTime.zoneName,
    }),
  )

  let start = performance.now()
  let runs = wasm.run(
    params.strategy,
    runsSpec.start,
    runsSpec.end,
    numMonths,
    asMFN(withdrawalStartMonth),
    annualToMonthlyReturnRate(params.returns.expectedAnnualReturns.stocks),
    annualToMonthlyReturnRate(params.returns.expectedAnnualReturns.bonds),
    Float64Array.from(
      params.returns.historicalMonthlyAdjusted.map((x) => x.stocks),
    ),
    Float64Array.from(
      params.returns.historicalMonthlyAdjusted.map((x) => x.bonds),
    ),
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
    params.byMonth.tpawAndSPAW.risk.lmp,
    params.byMonth.futureSavingsAndRetirementIncome.total,
    params.byMonth.adjustmentsToSpending.extraSpending.essential.total,
    params.byMonth.adjustmentsToSpending.extraSpending.discretionary.total,
    params.adjustmentsToSpending.tpawAndSPAW.legacy.target,
    params.adjustmentsToSpending.tpawAndSPAW.legacy.external,
    Float64Array.from(params.risk.tpawAndSPAW.monthlySpendingTilt),
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling ??
      undefined,
    params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor ?? undefined,
    params.sampling === 'monteCarlo'
      ? true
      : params.sampling === 'historical'
      ? false
      : noCase(params.sampling),
    params.samplingBlockSizeForMonteCarlo,
    MAX_AGE_IN_MONTHS,
    test?.truth ? Float64Array.from(test.truth) : undefined,
    test?.indexIntoHistoricalReturns
      ? Uint32Array.from(test.indexIntoHistoricalReturns)
      : undefined,
  )
  const perfRuns = performance.now() - start

  const numRuns = runsSpec.end - runsSpec.start
  const monthIndexes = _.range(0, numMonths)
  const splitArray = (x: Float64Array) => {
    const copy = x.slice()
    return monthIndexes.map((month) =>
      copy.subarray(month * numRuns, (month + 1) * numRuns),
    )
  }

  start = performance.now()

  const result: Omit<TPAWWorkerRunSimulationResult, 'perf'> = {
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
    averageAnnualReturns: {
      stocks: runs.average_annual_returns_stocks(),
      bonds: runs.average_annual_returns_bonds(),
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

// const getAnnualMeanFromMonthlyReturns = (monthlyReturns: Float64Array) =>
//   _.mean(
//     _.range(0, monthlyReturns.length - 12)
//       .map((i) => monthlyReturns.slice(i, i + 12))
//       .map(monthArrToYear),
//   )

// const monthArrToYear = (year: Float64Array) =>
//   year.map((x) => 1 + x).reduce(_.multiply, 1) - 1
