import _ from 'lodash'
import { fGet } from '../../Utils/Utils'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'

const _mergeNumberByWorkerByMonthsFromNowByRun = (
  numberByWorkerByMonthsFromNowByRun: Float64Array[][],
) => {
  const numMonths = numberByWorkerByMonthsFromNowByRun[0].length
  const numberByMonthsFromNowByWorkerByRun = _.times(
    numMonths,
    (x) => [] as Float64Array[],
  )

  for (const numberByMonthsFromNowByRun of numberByWorkerByMonthsFromNowByRun) {
    numberByMonthsFromNowByRun.forEach((x, i) =>
      numberByMonthsFromNowByWorkerByRun[i].push(x),
    )
  }

  const numberByMonthsFromNowByRun =
    numberByMonthsFromNowByWorkerByRun.map(_flattenTyped)

  return numberByMonthsFromNowByRun
}

export const mergeWorkerRuns = (
  runsByWorker: RunSimulationInWASMResult[],
): RunSimulationInWASMResult => {
  const byMonthsFromNowByRun = {
    savingsPortfolio: {
      start: {
        balance: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) => x.byMonthsFromNowByRun.savingsPortfolio.start.balance,
          ),
        ),
      },
      withdrawals: {
        essential: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) =>
              x.byMonthsFromNowByRun.savingsPortfolio.withdrawals.essential,
          ),
        ),
        discretionary: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) =>
              x.byMonthsFromNowByRun.savingsPortfolio.withdrawals.discretionary,
          ),
        ),
        regular: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) => x.byMonthsFromNowByRun.savingsPortfolio.withdrawals.regular,
          ),
        ),
        total: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) => x.byMonthsFromNowByRun.savingsPortfolio.withdrawals.total,
          ),
        ),
        fromSavingsPortfolioRate: _mergeNumberByWorkerByMonthsFromNowByRun(
          runsByWorker.map(
            (x) =>
              x.byMonthsFromNowByRun.savingsPortfolio.withdrawals
                .fromSavingsPortfolioRate,
          ),
        ),
      },
      afterWithdrawals: {
        allocation: {
          stocks: _mergeNumberByWorkerByMonthsFromNowByRun(
            runsByWorker.map(
              (x) =>
                x.byMonthsFromNowByRun.savingsPortfolio.afterWithdrawals
                  .allocation.stocks,
            ),
          ),
        },
      },
    },
    totalPortfolio: {
      afterWithdrawals: {
        allocation: {
          stocks: _mergeNumberByWorkerByMonthsFromNowByRun(
            runsByWorker.map(
              (x) =>
                x.byMonthsFromNowByRun.totalPortfolio.afterWithdrawals
                  .allocation.stocks,
            ),
          ),
        },
      },
    },
  }

  const byRun = {
    numInsufficientFundMonths: _flattenTypedI32(
      runsByWorker.map((s) => s.byRun.numInsufficientFundMonths),
    ),
    endingBalanceOfSavingsPortfolio: _flattenTyped(
      runsByWorker.map((s) => s.byRun.endingBalanceOfSavingsPortfolio),
    ),
  }
  const averageAnnualReturns = {
    stocks: _.mean(runsByWorker.map((x) => x.averageAnnualReturns.stocks)),
    bonds: _.mean(runsByWorker.map((x) => x.averageAnnualReturns.bonds)),
  }

  const worstPerf = fGet(
    _.last(_.sortBy(runsByWorker, (x) => x.perf[3][1])),
  ).perf
  return {
    byMonthsFromNowByRun,
    byRun,
    averageAnnualReturns,
    perf: worstPerf,
  }
}

const _flattenTyped = (arr: Float64Array[]) => {
  let offset = 0
  const result = new Float64Array(_.sumBy(arr, (x) => x.length))
  arr.forEach((x) => {
    result.set(x, offset)
    offset += x.length
  })
  return result
}
const _flattenTypedI32 = (arr: Int32Array[]) => {
  let offset = 0
  const result = new Int32Array(_.sumBy(arr, (x) => x.length))
  arr.forEach((x) => {
    result.set(x, offset)
    offset += x.length
  })
  return result
}
