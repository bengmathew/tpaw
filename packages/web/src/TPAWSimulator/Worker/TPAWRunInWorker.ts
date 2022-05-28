import _ from 'lodash'
import {nominalToReal} from '../../Utils/NominalToReal'
import {assert, fGet, noCase} from '../../Utils/Utils'
import {StatsTools} from '../StatsTools'
import {ValueForYearRange} from '../TPAWParams'
import {extendTPAWParams} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'
import {
  TPAWWorkerArgs,
  TPAWWorkerResult,
  TPAWWorkerRunSimulationResult,
} from './TPAWWorkerTypes'

export type TPAWRunInWorkerByPercentileByYearsFromNow = {
  byPercentileByYearsFromNow: {data: number[]; percentile: number}[]
}

export type TPAWRunInWorkerResult = {
  savingsPortfolio: {
    start: {
      balance: TPAWRunInWorkerByPercentileByYearsFromNow
    }
    withdrawals: {
      essential: {
        total: TPAWRunInWorkerByPercentileByYearsFromNow
        byId: Map<number, TPAWRunInWorkerByPercentileByYearsFromNow>
      }
      discretionary: {
        total: TPAWRunInWorkerByPercentileByYearsFromNow
        byId: Map<number, TPAWRunInWorkerByPercentileByYearsFromNow>
      }
      regular: TPAWRunInWorkerByPercentileByYearsFromNow
      total: TPAWRunInWorkerByPercentileByYearsFromNow
      fromSavingsPortfolioRate: TPAWRunInWorkerByPercentileByYearsFromNow
    }
    afterWithdrawals: {
      allocation: {
        stocks: TPAWRunInWorkerByPercentileByYearsFromNow
      }
    }
  }
  legacyByPercentile: {data: number; percentile: number}[]
  endingBalanceOfSavingsPortfolioByPercentile: {
    data: number
    percentile: number
  }[]
  firstYearOfSomeRun: {savingsPortfolio: SavingsPortfolioThroughAYear.End}
  perf: [
    ['workerRuns', number],
    ['mergeRunsByWorker', number],
    ['perfPostByYearsFromNow', number],
    ['perfPostByPercentile', number],
    ['total', number]
  ]
  perfByYearsFromNow: [
    ['pre', number],
    ['sort', number],
    ['post', number],
    ['percentile', number]
  ]

  perfByWorker: [
    ['runs', number],
    ['selectAndPivotPre', number],
    ['selectAndPivot', number],
    ['total', number]
  ]
}

export class TPAWRunInWorker {
  private _workers: Worker[]
  private _resolvers = {
    runSimulation: new Map<
      string,
      (value: TPAWWorkerRunSimulationResult) => void
    >(),
    sortRows: new Map<string, (value: Float64Array[]) => void>(),
  }
  constructor() {
    const numWorkers = navigator.hardwareConcurrency || 4
    this._workers = _.range(numWorkers).map(
      () => new Worker(new URL('./TPAWWorker.ts', import.meta.url))
    )

    this._workers.forEach(
      worker =>
        (worker.onmessage = event => {
          const {taskID, ...data}: TPAWWorkerResult = event.data
          switch (data.type) {
            case 'runSimulation': {
              const resolve = fGet(this._resolvers.runSimulation.get(taskID))
              this._resolvers.runSimulation.delete(taskID)
              resolve(data.result)
              break
            }
            case 'sortRows': {
              const resolve = fGet(this._resolvers.sortRows.get(taskID))
              this._resolvers.sortRows.delete(taskID)
              resolve(data.result)
              break
            }
            default:
              noCase(data)
          }
        })
    )
  }

  private _runSimulation(
    worker: Worker,
    numRuns: number,
    params: TPAWParamsProcessed
  ): Promise<TPAWWorkerRunSimulationResult> {
    const taskID = _.uniqueId()
    const args: Extract<TPAWWorkerArgs, {type: 'runSimulation'}>['args'] = {
      numRuns,
      params,
    }
    worker.postMessage({taskID, type: 'runSimulation', args})
    return new Promise<TPAWWorkerRunSimulationResult>(resolve =>
      this._resolvers.runSimulation.set(taskID, resolve)
    )
  }
  private _sortRows(
    worker: Worker,
    data: Float64Array[]
  ): Promise<Float64Array[]> {
    const taskID = _.uniqueId()
    const transferables: Transferable[] = data.map(x => x.buffer)
    const args: Extract<TPAWWorkerArgs, {type: 'sortRows'}>['args'] = {
      data,
    }
    worker.postMessage({taskID, type: 'sortRows', args}, transferables)
    return new Promise<Float64Array[]>(resolve =>
      this._resolvers.sortRows.set(taskID, resolve)
    )
  }

  async runSimulations(
    status: {canceled: boolean},
    numRuns: number,
    params: TPAWParamsProcessed,
    percentiles: number[]
  ): Promise<TPAWRunInWorkerResult | null> {
    const start0 = performance.now()
    let start = performance.now()
    const runsByWorker = await Promise.all(
      this._workers.map((worker, i) =>
        this._runSimulation(
          worker,
          _loadBalance(i, numRuns, this._workers.length).length,
          params
        )
      )
    )
    const perfWorkerRuns = performance.now() - start
    start = performance.now()

    if (status.canceled) return null
    const {
      byYearsFromNowByRun,
      legacyByRun,
      endingBalanceOfSavingsPortfolioByRun,
      firstYearOfSomeRun,
      perf: perfByWorker,
    } = _mergeRunsByWorker(runsByWorker)

    const perfMergeRunsByWorker = performance.now() - start
    start = performance.now()

    const perfByYearsFromNow = {
      pre: 0,
      sort: 0,
      post: 0,
      percentile: 0,
    }

    const _processForByPercentileByYearsFromNow = async (
      numberByTypeByYearsFromNowByRun: Float64Array[][]
    ) => {
      let start = performance.now()
      const numYearsArr = _.uniq(
        numberByTypeByYearsFromNowByRun.map(x => x.length)
      )
      assert(numYearsArr.length === 1)
      const numYears = numYearsArr[0]
      const numberByYearsFromNowByRun = _.flatten(
        numberByTypeByYearsFromNowByRun
      )
      perfByYearsFromNow.pre = performance.now() - start
      start = performance.now()
      const numberByWorkerByYearsFromNowByRunSorted = await Promise.all(
        this._workers.map((worker, i) => {
          const {start, length} = _loadBalance(
            i,
            numberByYearsFromNowByRun.length,
            this._workers.length
          )
          return this._sortRows(
            worker,
            numberByYearsFromNowByRun.slice(start, start + length)
          )
        })
      )
      perfByYearsFromNow.sort = performance.now() - start
      start = performance.now()

      const numberByYearsFromNowByRunSorted = _.flatten(
        numberByWorkerByYearsFromNowByRunSorted
      )

      perfByYearsFromNow.post = performance.now() - start
      start = performance.now()

      const numberByTypeByYearsFromNowByRunSorted =
        numberByTypeByYearsFromNowByRun.map((x, i) =>
          numberByYearsFromNowByRunSorted.slice(
            i * numYears,
            (i + 1) * numYears
          )
        )
      const result = numberByTypeByYearsFromNowByRunSorted.map(
        numberByYearsFromNowByRunSorted => {
          const numberByYearsFromNowByPercentile =
            StatsTools.pickPercentilesFromSorted(
              numberByYearsFromNowByRunSorted,
              percentiles
            )

          const byPercentileByYearsFromNow = StatsTools.pivot(
            numberByYearsFromNowByPercentile
          ).map((data, i) => ({data, percentile: percentiles[i]}))

          return {byPercentileByYearsFromNow}
        }
      )

      perfByYearsFromNow.percentile = performance.now() - start
      start = performance.now()
      return result
    }

    const _processForByPercentile = async (numberByRun: Float64Array) => {
      numberByRun.sort()

      return StatsTools.pickPercentilesFromSorted(
        [numberByRun],
        percentiles
      )[0].map((data, i) => ({data, percentile: percentiles[i]}))
    }

    const [
      startingBalanceOfSavingsPortfolio,
      withdrawalsEssential,
      withdrawalsDiscretionary,
      withdrawalsRegular,
      withdrawalsTotal,
      withdrawalFromSavingsRate,
      savingsPortfolioStockAllocation,
    ] = await _processForByPercentileByYearsFromNow([
      byYearsFromNowByRun.savingsPortfolio.start.balance,
      byYearsFromNowByRun.savingsPortfolio.withdrawals.essential,
      byYearsFromNowByRun.savingsPortfolio.withdrawals.discretionary,
      byYearsFromNowByRun.savingsPortfolio.withdrawals.regular,
      byYearsFromNowByRun.savingsPortfolio.withdrawals.total,
      byYearsFromNowByRun.savingsPortfolio.withdrawals.fromSavingsPortfolioRate,
      byYearsFromNowByRun.savingsPortfolio.afterWithdrawals.allocation.stocks,
    ])

    if (status.canceled) return null

    const withdrawlsEssentialById = new Map(
      params.withdrawals.essential.map(x => [
        x.id,
        _separateExtraWithdrawal(x, params, withdrawalsEssential, 'essential'),
      ])
    )
    const withdrawlsDiscretionaryById = new Map(
      params.withdrawals.discretionary.map(x => [
        x.id,
        _separateExtraWithdrawal(
          x,
          params,
          withdrawalsDiscretionary,
          'discretionary'
        ),
      ])
    )

    const perfPostByYearsFromNow = performance.now() - start
    start = performance.now()
    const legacyByPercentile = await _processForByPercentile(legacyByRun)
    if (status.canceled) return null
    const endingBalanceOfSavingsPortfolioByPercentile =
      await _processForByPercentile(endingBalanceOfSavingsPortfolioByRun)
    if (status.canceled) return null

    const perfPostByPercentile = performance.now() - start
    const result: TPAWRunInWorkerResult = {
      savingsPortfolio: {
        start: {
          balance: startingBalanceOfSavingsPortfolio,
        },
        withdrawals: {
          essential: {
            total: withdrawalsEssential,
            byId: withdrawlsEssentialById,
          },
          discretionary: {
            total: withdrawalsDiscretionary,
            byId: withdrawlsDiscretionaryById,
          },
          regular: withdrawalsRegular,
          total: withdrawalsTotal,
          fromSavingsPortfolioRate: _mapByPercentileByYearsFromNow(
            withdrawalFromSavingsRate,
            x => Math.max(0, x)
          ),
        },
        afterWithdrawals: {
          allocation: {
            stocks: savingsPortfolioStockAllocation,
          },
        },
      },

      firstYearOfSomeRun,
      legacyByPercentile,
      endingBalanceOfSavingsPortfolioByPercentile,
      perf: [
        ['workerRuns', perfWorkerRuns],
        ['mergeRunsByWorker', perfMergeRunsByWorker],
        ['perfPostByYearsFromNow', perfPostByYearsFromNow],
        ['perfPostByPercentile', perfPostByPercentile],
        ['total', performance.now() - start0],
      ],
      perfByYearsFromNow: [
        ['pre', perfByYearsFromNow.pre],
        ['sort', perfByYearsFromNow.sort],
        ['post', perfByYearsFromNow.post],
        ['percentile', perfByYearsFromNow.percentile],
      ],
      perfByWorker,
    }
    return status.canceled ? null : result
  }

  terminate() {
    this._workers.forEach(x => x.terminate())
  }
}
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

const _mergeRunsByWorker = (
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

  const firstYearOfSomeRun = runsByWorker[0].firstYearOfSomeRun

  const legacyByRun = _flattenTyped(runsByWorker.map(s => s.legacyByRun))
  const endingBalanceOfSavingsPortfolioByRun = _flattenTyped(
    runsByWorker.map(s => s.endingBalanceOfSavingsPortfolioByRun)
  )
  return {
    byYearsFromNowByRun,
    firstYearOfSomeRun,
    legacyByRun,
    endingBalanceOfSavingsPortfolioByRun,
    perf: [
      ['runs', Math.max(...runsByWorker.map(x => x.perf[0][1]))],
      ['selectAndPivotPre', Math.max(...runsByWorker.map(x => x.perf[1][1]))],
      ['selectAndPivot', Math.max(...runsByWorker.map(x => x.perf[2][1]))],
      ['total', Math.max(...runsByWorker.map(x => x.perf[3][1]))],
    ],
  }
}

export const _loadBalance = (
  worker: number,
  numJobs: number,
  numWorkers: number
) => {
  const remainder = numJobs % numWorkers
  const minJobsPerWorker = Math.floor(numJobs / numWorkers)
  const start = worker * minJobsPerWorker + Math.min(remainder, worker)
  const length = minJobsPerWorker + (worker < remainder ? 1 : 0)
  return {start, length}
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

const _separateExtraWithdrawal = (
  discretionaryWithdrawal: ValueForYearRange,
  params: TPAWParamsProcessed,
  x: TPAWRunInWorkerByPercentileByYearsFromNow,
  type: 'discretionary' | 'essential'
): TPAWRunInWorkerByPercentileByYearsFromNow => {
  const yearRange = extendTPAWParams(params.original).asYFN(
    discretionaryWithdrawal.yearRange
  )

  return _mapByPercentileByYearsFromNow(x, (value, yearFromNow) => {
    if (yearFromNow < yearRange.start || yearFromNow > yearRange.end) {
      return 0
    }
    const currYearParams = params.byYear[yearFromNow].withdrawals
    const withdrawalTargetForThisYear = nominalToReal(
      discretionaryWithdrawal,
      params.original.inflation,
      yearFromNow
    )
    if (withdrawalTargetForThisYear === 0) return 0
    const ratio =
      withdrawalTargetForThisYear /
      (type === 'discretionary'
        ? currYearParams.discretionary
        : currYearParams.essential)
    assert(!isNaN(ratio)) // withdrawalTargetForThisYear ?>0 imples denominator is not 0.
    return value * ratio
  })
}

const _mapByPercentileByYearsFromNow = (
  x: TPAWRunInWorkerByPercentileByYearsFromNow,
  fn: (x: number, i: number) => number
): TPAWRunInWorkerByPercentileByYearsFromNow => {
  const byPercentileByYearsFromNow = x.byPercentileByYearsFromNow.map(x => ({
    data: x.data.map(fn),
    percentile: x.percentile,
  }))
  return {byPercentileByYearsFromNow}
}
