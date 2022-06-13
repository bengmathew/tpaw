import _ from 'lodash'
import {nominalToReal} from '../../Utils/NominalToReal'
import {assert, fGet, noCase} from '../../Utils/Utils'
import {StatsTools} from '../StatsTools'
import {ValueForYearRange} from '../TPAWParams'
import {extendTPAWParams} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {
  FirstYearSavingsPortfolioDetail,
  firstYearSavingsPortfolioDetail,
} from './FirstYearSavingsPortfolioDetail'
import {mergeWorkerRuns} from './MergeWorkerRuns'
import {
  TPAWWorkerArgs,
  TPAWWorkerCalculateOneOverCVResult,
  TPAWWorkerResult,
  TPAWWorkerRunSimulationResult,
  TPAWWorkerSortResult,
} from './TPAWWorkerTypes'

export type TPAWRunInWorkerByPercentileByYearsFromNow = {
  byPercentileByYearsFromNow: {data: number[]; percentile: number}[]
}

const MULTI_THREADED = true

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
    sharpeRatio: {
      withdrawals: {
        regular: Float64Array
      }
    }
    afterWithdrawals: {
      allocation: {
        stocks: TPAWRunInWorkerByPercentileByYearsFromNow
      }
    }
  }
  endingBalanceOfSavingsPortfolioByPercentile: {
    data: number
    percentile: number
  }[]
  firstYearOfSomeRun: FirstYearSavingsPortfolioDetail
  perf: {
    main: [
      ['num-workers', number],
      ['simulation', number],
      ['merge-simulation', number],
      ['sort-and-pick-percentiles', number],
      ['post', number],
      ['rest', number],
      ['total', number]
    ]
    sortAndPickPercentilesYearly: [
      ['pre', number],
      ['slowest-sort-worker', number],
      ['sort', number],
      ['post', number],
      ['percentile', number],
      ['rest', number],
      ['total', number]
    ]
    slowestSimulationWorker: [
      ['runs', number],
      ['post', number],
      ['rest', number],
      ['total', number]
    ]
  }
}

export class TPAWRunInWorker {
  private _workers: Worker[]
  private _resolvers = {
    runSimulation: new Map<
      string,
      (value: TPAWWorkerRunSimulationResult) => void
    >(),
    sort: new Map<string, (value: TPAWWorkerSortResult) => void>(),
    calculateOneOverCV: new Map<
      string,
      (value: TPAWWorkerCalculateOneOverCVResult) => void
    >(),
  }
  constructor() {
    const numWorkers = MULTI_THREADED ? navigator.hardwareConcurrency || 4 : 1
    this._workers = _.range(numWorkers).map(
      () => new Worker(new URL('./TPAWWorker.ts', import.meta.url))
    )

    this._workers.forEach(
      worker =>
        (worker.onmessage = event => {
          const data = event.data as TPAWWorkerResult
          const {taskID} = data
          switch (data.type) {
            case 'runSimulation': {
              const resolve = fGet(this._resolvers.runSimulation.get(taskID))
              this._resolvers.runSimulation.delete(taskID)
              resolve(data.result)
              break
            }
            case 'sort': {
              const resolve = fGet(this._resolvers.sort.get(taskID))
              this._resolvers.sort.delete(taskID)
              resolve(data.result)
              break
            }
            case 'calculateOneOverCV': {
              const resolve = fGet(
                this._resolvers.calculateOneOverCV.get(taskID)
              )
              this._resolvers.calculateOneOverCV.delete(taskID)
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
    const message: Extract<TPAWWorkerArgs, {type: 'runSimulation'}> = {
      taskID,
      type: 'runSimulation',
      args: {numRuns, params},
    }
    worker.postMessage(message)
    return new Promise<TPAWWorkerRunSimulationResult>(resolve =>
      this._resolvers.runSimulation.set(taskID, resolve)
    )
  }
  private _sort(
    worker: Worker,
    data: Float64Array[]
  ): Promise<TPAWWorkerSortResult> {
    const taskID = _.uniqueId()
    const transferables: Transferable[] = data.map(x => x.buffer)
    const message: Extract<TPAWWorkerArgs, {type: 'sort'}> = {
      taskID,
      type: 'sort',
      args: {data},
    }
    worker.postMessage(message, transferables)
    return new Promise<TPAWWorkerSortResult>(resolve =>
      this._resolvers.sort.set(taskID, resolve)
    )
  }

  private _calculateOneOverCV(
    worker: Worker,
    data: Float64Array[]
  ): Promise<TPAWWorkerCalculateOneOverCVResult> {
    const taskID = _.uniqueId()
    const transferables: Transferable[] = data.map(x => x.buffer)
    const message: Extract<TPAWWorkerArgs, {type: 'calculateOneOverCV'}> = {
      taskID,
      type: 'calculateOneOverCV',
      args: {data},
    }
    worker.postMessage(message, transferables)
    return new Promise<TPAWWorkerCalculateOneOverCVResult>(resolve =>
      this._resolvers.calculateOneOverCV.set(taskID, resolve)
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
    const perfSimulation = performance.now() - start
    start = performance.now()

    if (status.canceled) return null
    const {
      byYearsFromNowByRun,
      byRun,
      perf: perfSlowestSimulationWorker,
    } = mergeWorkerRuns(runsByWorker)

    const perfMergeSimulation = performance.now() - start
    start = performance.now()

    const {
      result: [
        startingBalanceOfSavingsPortfolio,
        withdrawalsEssential,
        withdrawalsDiscretionary,
        withdrawalsRegular,
        withdrawalsTotal,
        withdrawalFromSavingsRate,
        savingsPortfolioStockAllocation,
      ],
      perf: perfSortAndPickPercentilesYearly,
    } = await this._sortAndPickPercentilesForByYearsFromNowByRun(
      [
        byYearsFromNowByRun.savingsPortfolio.start.balance,
        byYearsFromNowByRun.savingsPortfolio.withdrawals.essential,
        byYearsFromNowByRun.savingsPortfolio.withdrawals.discretionary,
        byYearsFromNowByRun.savingsPortfolio.withdrawals.regular,
        byYearsFromNowByRun.savingsPortfolio.withdrawals.total,
        byYearsFromNowByRun.savingsPortfolio.withdrawals
          .fromSavingsPortfolioRate,
        byYearsFromNowByRun.savingsPortfolio.afterWithdrawals.allocation.stocks,
      ],
      percentiles
    )

    if (status.canceled) return null

    const endingBalanceOfSavingsPortfolioByPercentile =
      _sortAndPickPercentilesForByRun(
        byRun.endingBalanceOfSavingsPortfolio,
        percentiles
      )
    if (status.canceled) return null

    const perfSortAndPickPercentiles = performance.now() - start
    start = performance.now()

    const {data: sharpeRatioWithdrawalsRegular} =
      await this._calculateOneOverCV(
        this._workers[0],
        byYearsFromNowByRun.savingsPortfolio.excessWithdrawals.regular
      )

    const firstYearOfSomeRun = firstYearSavingsPortfolioDetail(
      runsByWorker[0].byYearsFromNowByRun.savingsPortfolio,
      params
    )
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

    const perfPost = performance.now() - start
    start = performance.now()

    const perfTotal = performance.now() - start0
    const perfRest =
      perfTotal -
      perfSimulation -
      perfMergeSimulation -
      perfSortAndPickPercentiles -
      perfPost
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

        sharpeRatio: {withdrawals: {regular: sharpeRatioWithdrawalsRegular}},

        afterWithdrawals: {
          allocation: {
            stocks: savingsPortfolioStockAllocation,
          },
        },
      },
      firstYearOfSomeRun,
      endingBalanceOfSavingsPortfolioByPercentile,
      perf: {
        main: [
          ['num-workers', this._workers.length],
          ['simulation', perfSimulation],
          ['merge-simulation', perfMergeSimulation],
          ['sort-and-pick-percentiles', perfSortAndPickPercentiles],
          ['post', perfPost],
          ['rest', perfRest],
          ['total', perfTotal],
        ],
        sortAndPickPercentilesYearly: [
          ['pre', perfSortAndPickPercentilesYearly.pre],
          [
            'slowest-sort-worker',
            perfSortAndPickPercentilesYearly.slowestSortWorker,
          ],
          ['sort', perfSortAndPickPercentilesYearly.sort],
          ['post', perfSortAndPickPercentilesYearly.post],
          ['percentile', perfSortAndPickPercentilesYearly.percentile],
          ['rest', perfSortAndPickPercentilesYearly.rest],
          ['total', perfSortAndPickPercentilesYearly.total],
        ],
        slowestSimulationWorker: perfSlowestSimulationWorker,
      },
    }
    return status.canceled ? null : result
  }

  private _sortAndPickPercentilesForByYearsFromNowByRun = async (
    numberByTypeByYearsFromNowByRun: Float64Array[][],
    percentiles: number[]
  ) => {
    const perf = {
      pre: 0,
      slowestSortWorker: 0,
      sort: 0,
      post: 0,
      percentile: 0,
      rest: 0,
      total: 0,
    }
    const start0 = performance.now()
    let start = performance.now()
    const numYearsArr = _.uniq(
      numberByTypeByYearsFromNowByRun.map(x => x.length)
    )
    assert(numYearsArr.length === 1)
    const numYears = numYearsArr[0]
    const numberByYearsFromNowByRun = _.flatten(numberByTypeByYearsFromNowByRun)
    perf.pre = performance.now() - start
    start = performance.now()
    const numberByWorkerByYearsFromNowByRunSorted = await Promise.all(
      this._workers.map((worker, i) => {
        const {start, length} = _loadBalance(
          i,
          numberByYearsFromNowByRun.length,
          this._workers.length
        )
        return this._sort(
          worker,
          numberByYearsFromNowByRun.slice(start, start + length)
        )
      })
    )
    perf.slowestSortWorker = Math.max(
      ...numberByWorkerByYearsFromNowByRunSorted.map(x => x.perf)
    )
    perf.sort = performance.now() - start
    start = performance.now()

    const numberByYearsFromNowByRunSorted = _.flatten(
      numberByWorkerByYearsFromNowByRunSorted.map(x => x.data)
    )

    perf.post = performance.now() - start
    start = performance.now()

    const numberByTypeByYearsFromNowByRunSorted =
      numberByTypeByYearsFromNowByRun.map((x, i) =>
        numberByYearsFromNowByRunSorted.slice(i * numYears, (i + 1) * numYears)
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

    perf.percentile = performance.now() - start
    perf.total = performance.now() - start0
    perf.rest =
      perf.total - (perf.pre + perf.sort + perf.post + perf.percentile)

    return {result, perf}
  }

  terminate() {
    this._workers.forEach(x => x.terminate())
  }
}

const _sortAndPickPercentilesForByRun = (
  numberByRun: Float64Array,
  percentiles: number[]
) => {
  numberByRun.sort()
  return StatsTools.pickPercentilesFromSorted(
    [numberByRun],
    percentiles
  )[0].map((data, i) => ({data, percentile: percentiles[i]}))
}

const _loadBalance = (worker: number, numJobs: number, numWorkers: number) => {
  const remainder = numJobs % numWorkers
  const minJobsPerWorker = Math.floor(numJobs / numWorkers)
  const start = worker * minJobsPerWorker + Math.min(remainder, worker)
  const length = minJobsPerWorker + (worker < remainder ? 1 : 0)
  return {start, length}
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
