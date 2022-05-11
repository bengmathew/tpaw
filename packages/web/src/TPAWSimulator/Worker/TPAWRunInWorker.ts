import _ from 'lodash'
import {nominalToReal} from '../../Utils/NominalToReal'
import {assert, fGet, noCase} from '../../Utils/Utils'
import {TPAWSimulationForYear} from '../RunTPAWSimulation'
import {StatsTools} from '../StatsTools'
import {ValueForYearRange} from '../TPAWParams'
import {extendTPAWParams} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {
  TPAWWorkerArgs,
  TPAWWorkerResult,
  TPAWWorkerRunSimulationResult,
} from './TPAWWorkerTypes'

export type TPAWRunInWorkerByPercentileByYearsFromNow = {
  byPercentileByYearsFromNow: {data: number[]; percentile: number}[]
}

export type TPAWRunInWorkerResult = {
  withdrawals: {
    total: TPAWRunInWorkerByPercentileByYearsFromNow
    essential: {
      total: TPAWRunInWorkerByPercentileByYearsFromNow
      byId: Map<number, TPAWRunInWorkerByPercentileByYearsFromNow>
    }
    extra: {
      total: TPAWRunInWorkerByPercentileByYearsFromNow
      byId: Map<number, TPAWRunInWorkerByPercentileByYearsFromNow>
    }
    regular: TPAWRunInWorkerByPercentileByYearsFromNow
  }
  startingBalanceOfSavingsPortfolio: TPAWRunInWorkerByPercentileByYearsFromNow
  savingsPortfolioStockAllocation: TPAWRunInWorkerByPercentileByYearsFromNow
  withdrawalFromSavingsRate: TPAWRunInWorkerByPercentileByYearsFromNow
  legacyByPercentile: {data: number; percentile: number}[]
  endingBalanceOfSavingsPortfolioByPercentile: {
    data: number
    percentile: number
  }[]
  firstYearOfSomeRun: TPAWSimulationForYear
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
      withdrawalsTotal,
      withdrawalsEssential,
      withdrawalsExtra,
      withdrawalsRegular,
      startingBalanceOfSavingsPortfolio,
      savingsPortfolioStockAllocation,
      withdrawalFromSavingsRate,
    ] = await _processForByPercentileByYearsFromNow([
      byYearsFromNowByRun.withdrawals.total,
      byYearsFromNowByRun.withdrawals.essential,
      byYearsFromNowByRun.withdrawals.extra,
      byYearsFromNowByRun.withdrawals.regular,
      byYearsFromNowByRun.startingBalanceOfSavingsPortfolio,
      byYearsFromNowByRun.savingsPortfolioStockAllocation,
      byYearsFromNowByRun.withdrawalFromSavingsRate,
    ])

    if (status.canceled) return null

    const withdrawlsEssentialById = new Map(
      params.withdrawals.fundedByBonds.map(x => [
        x.id,
        _separateExtraWithdrawal(x, params, withdrawalsEssential, 'essential'),
      ])
    )
    const withdrawlsExtraById = new Map(
      params.withdrawals.fundedByRiskPortfolio.map(x => [
        x.id,
        _separateExtraWithdrawal(x, params, withdrawalsExtra, 'extra'),
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
      withdrawals: {
        total: withdrawalsTotal,
        essential: {total: withdrawalsEssential, byId: withdrawlsEssentialById},
        extra: {total: withdrawalsExtra, byId: withdrawlsExtraById},
        regular: withdrawalsRegular,
      },
      startingBalanceOfSavingsPortfolio,
      savingsPortfolioStockAllocation,
      withdrawalFromSavingsRate,
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
    withdrawals: {
      total: _mergeNumberByWorkerByYearsFromNowByRun(
        runsByWorker.map(x => x.byYearsFromNowByRun.withdrawals.total)
      ),
      essential: _mergeNumberByWorkerByYearsFromNowByRun(
        runsByWorker.map(x => x.byYearsFromNowByRun.withdrawals.essential)
      ),
      extra: _mergeNumberByWorkerByYearsFromNowByRun(
        runsByWorker.map(x => x.byYearsFromNowByRun.withdrawals.extra)
      ),
      regular: _mergeNumberByWorkerByYearsFromNowByRun(
        runsByWorker.map(x => x.byYearsFromNowByRun.withdrawals.regular)
      ),
    },
    startingBalanceOfSavingsPortfolio: _mergeNumberByWorkerByYearsFromNowByRun(
      runsByWorker.map(
        x => x.byYearsFromNowByRun.startingBalanceOfSavingsPortfolio
      )
    ),
    savingsPortfolioStockAllocation: _mergeNumberByWorkerByYearsFromNowByRun(
      runsByWorker.map(
        x => x.byYearsFromNowByRun.savingsPortfolioStockAllocation
      )
    ),
    withdrawalFromSavingsRate: _mergeNumberByWorkerByYearsFromNowByRun(
      runsByWorker.map(x => x.byYearsFromNowByRun.withdrawalFromSavingsRate)
    ),
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
  extraWithdrawal: ValueForYearRange,
  params: TPAWParamsProcessed,
  x: TPAWRunInWorkerByPercentileByYearsFromNow,
  type: 'extra' | 'essential'
): TPAWRunInWorkerByPercentileByYearsFromNow => {
  const yearRange = extendTPAWParams(params.original).asYFN(
    extraWithdrawal.yearRange
  )

  const byPercentileByYearsFromNow = x.byPercentileByYearsFromNow.map(
    ({data, percentile}) => ({
      data: data.map((value, yearFromNow) => {
        if (yearFromNow < yearRange.start || yearFromNow > yearRange.end) {
          return 0
        }
        const currYearParams = params.byYear[yearFromNow].withdrawals
        const withdrawalTargetForThisYear = nominalToReal(
          extraWithdrawal,
          params.original.inflation,
          yearFromNow
        )
        if (withdrawalTargetForThisYear === 0) return 0
        const ratio =
          withdrawalTargetForThisYear /
          (type === 'extra'
            ? currYearParams.fundedByRiskPortfolio
            : currYearParams.fundedByBonds)
        assert(!isNaN(ratio)) // withdrawalTargetForThisYear ?>0 imples denominator is not 0.
        return value * ratio
      }),
      percentile,
    })
  )
  return {byPercentileByYearsFromNow}
}
