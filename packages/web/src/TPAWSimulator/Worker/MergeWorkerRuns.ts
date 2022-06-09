import _ from 'lodash'
import {fGet} from '../../Utils/Utils'
import {TPAWWorkerRunSimulationResult} from './TPAWWorkerTypes'

const _mergeNumberByWorkerByYearsFromNowByRun = (
  numberByWorkerByYearsFromNowByRun: Float64Array[][]
) => {
  const numYears = numberByWorkerByYearsFromNowByRun[0].length
  const numberByYearsFromNowByWorkerByRun = _.times(
    numYears,
    x => [] as Float64Array[]
  )

  for (const numberByYearsFromNowByRun of numberByWorkerByYearsFromNowByRun) {
    numberByYearsFromNowByRun.forEach((x, i) =>
      numberByYearsFromNowByWorkerByRun[i].push(x)
    )
  }

  const numberByYearsFromNowByRun =
    numberByYearsFromNowByWorkerByRun.map(_flattenTyped)

  return numberByYearsFromNowByRun
}

export const mergeWorkerRuns = (
  runsByWorker: TPAWWorkerRunSimulationResult[]
): TPAWWorkerRunSimulationResult => {
  const byYearsFromNowByRun = {
    savingsPortfolio: {
      start: {
        balance: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x => x.byYearsFromNowByRun.savingsPortfolio.start.balance
          )
        ),
      },
      withdrawals: {
        essential: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x => x.byYearsFromNowByRun.savingsPortfolio.withdrawals.essential
          )
        ),
        discretionary: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x =>
              x.byYearsFromNowByRun.savingsPortfolio.withdrawals.discretionary
          )
        ),
        regular: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x => x.byYearsFromNowByRun.savingsPortfolio.withdrawals.regular
          )
        ),
        total: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x => x.byYearsFromNowByRun.savingsPortfolio.withdrawals.total
          )
        ),
        fromSavingsPortfolioRate: _mergeNumberByWorkerByYearsFromNowByRun(
          runsByWorker.map(
            x =>
              x.byYearsFromNowByRun.savingsPortfolio.withdrawals
                .fromSavingsPortfolioRate
          )
        ),
      },
      afterWithdrawals: {
        allocation: {
          stocks: _mergeNumberByWorkerByYearsFromNowByRun(
            runsByWorker.map(
              x =>
                x.byYearsFromNowByRun.savingsPortfolio.afterWithdrawals
                  .allocation.stocks
            )
          ),
        },
      },
    },
  }

  const byRun = {
    endingBalanceOfSavingsPortfolio: _flattenTyped(
      runsByWorker.map(s => s.byRun.endingBalanceOfSavingsPortfolio)
    ),
  }

  const worstPerf = fGet(_.last(_.sortBy(runsByWorker, x => x.perf[3][1]))).perf
  return {
    byYearsFromNowByRun,
    byRun,
    perf: worstPerf,
  }
}

const _flattenTyped = (arr: Float64Array[]) => {
  let offset = 0
  const result = new Float64Array(_.sumBy(arr, x => x.length))
  arr.forEach(x => {
    result.set(x, offset)
    offset += x.length
  })
  return result
}
