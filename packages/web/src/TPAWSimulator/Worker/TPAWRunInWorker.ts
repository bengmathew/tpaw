import _ from 'lodash'
import {fGet, noCase} from '../../Utils/Utils'
import {TPAWSimulationForYear} from '../RunTPAWSimulation'
import {StatsTools} from '../StatsTools'
import {TPAWParams} from '../TPAWParams'
import {
  TPAWWorkerResult,
  TPAWWorkerRunSimulationResult,
} from './TPAWWorkerTypes'

export type TPAWRunInWorkerByPercentileByYearsFromNow = {
  byPercentileByYearsFromNow: {data: number[]; percentile: number}[]
}

export type TPAWRunInWorkerResult = {
  withdrawals: {
    total: TPAWRunInWorkerByPercentileByYearsFromNow
    essential: TPAWRunInWorkerByPercentileByYearsFromNow
    extra: TPAWRunInWorkerByPercentileByYearsFromNow
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
}

export class TPAWRunInWorker {
  private _workers: Worker[]
  private _resolvers = {
    runSimulation: new Map<
      string,
      (value: TPAWWorkerRunSimulationResult) => void
    >(),
    sortRows: new Map<string, (value: number[][]) => void>(),
  }
  constructor() {
    // TODO:
    // const numWorkers = navigator.hardwareConcurrency || 4
    const numWorkers = 1
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
    params: TPAWParams
  ): Promise<TPAWWorkerRunSimulationResult> {
    const taskID = _.uniqueId()
    worker.postMessage({taskID, type: 'runSimulation', args: {numRuns, params}})
    return new Promise<TPAWWorkerRunSimulationResult>(resolve =>
      this._resolvers.runSimulation.set(taskID, resolve)
    )
  }
  private _sortRows(worker: Worker, data: number[][]): Promise<number[][]> {
    const taskID = _.uniqueId()
    worker.postMessage({taskID, type: 'sortRows', args: {data}})
    return new Promise<number[][]>(resolve =>
      this._resolvers.sortRows.set(taskID, resolve)
    )
  }

  async runSimulations(
    status: {canceled: boolean},
    numRuns: number,
    params: TPAWParams,
    percentiles: number[]
  ): Promise<TPAWRunInWorkerResult | null> {
    const runsByWorker = await Promise.all(
      this._workers.map((worker, i) =>
        this._runSimulation(
          worker,
          _loadBalance(i, numRuns, this._workers.length).length,
          params
        )
      )
    )

    if (status.canceled) return null
    const {
      byYearsFromNowByRun,
      legacyByRun,
      endingBalanceOfSavingsPortfolioByRun,
      firstYearOfSomeRun,
    } =  runsByWorker[0]
    _mergeRunsByWorker(runsByWorker)

    const _processForByPercentileByYearsFromNow = async (
      numberByYearsFromNowByRun: number[][]
    ) => {
      const numberByWorkerByYearsFromNowByRunSorted = await Promise.all(
        this._workers.map((worker, i) => {
          const {start, length} = _loadBalance(i, numRuns, this._workers.length)
          return this._sortRows(
            worker,
            numberByYearsFromNowByRun.slice(start, start + length)
          )
        })
      )

      if (status.canceled) return null
      const numberByYearsFromNowByRunSorted = _.flatten(
        numberByWorkerByYearsFromNowByRunSorted
      )
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

    const _processForByPercentile = async (numberByRun: number[]) => {
      const numberByRunSorted = (
        await this._sortRows(this._workers[0], [numberByRun])
      )[0]

      return StatsTools.pickPercentilesFromSorted(
        [numberByRunSorted],
        percentiles
      )[0].map((data, i) => ({data, percentile: percentiles[i]}))
    }

    const withdrawalsTotal = await _processForByPercentileByYearsFromNow(
      byYearsFromNowByRun.withdrawals.total
    )
    if (withdrawalsTotal === null) return null
    const withdrawalsEssential = await _processForByPercentileByYearsFromNow(
      byYearsFromNowByRun.withdrawals.essential
    )
    if (withdrawalsEssential === null) return null
    const withdrawalsExtra = await _processForByPercentileByYearsFromNow(
      byYearsFromNowByRun.withdrawals.extra
    )
    if (withdrawalsExtra === null) return null
    const withdrawalsRegular = await _processForByPercentileByYearsFromNow(
      byYearsFromNowByRun.withdrawals.regular
    )
    if (withdrawalsRegular === null) return null
    const startingBalanceOfSavingsPortfolio =
      await _processForByPercentileByYearsFromNow(
        byYearsFromNowByRun.startingBalanceOfSavingsPortfolio
      )
    if (startingBalanceOfSavingsPortfolio === null) return null
    const savingsPortfolioStockAllocation =
      await _processForByPercentileByYearsFromNow(
        byYearsFromNowByRun.savingsPortfolioStockAllocation
      )
    if (savingsPortfolioStockAllocation === null) return null
    const withdrawalFromSavingsRate =
      await _processForByPercentileByYearsFromNow(
        byYearsFromNowByRun.withdrawalFromSavingsRate
      )
    if (withdrawalFromSavingsRate === null) return null

    const legacyByPercentile = await _processForByPercentile(legacyByRun)
    if (status.canceled) return null
    const endingBalanceOfSavingsPortfolioByPercentile =
      await _processForByPercentile(endingBalanceOfSavingsPortfolioByRun)
    if (status.canceled) return null

    const result = {
      withdrawals: {
        total: withdrawalsTotal,
        essential: withdrawalsEssential,
        extra: withdrawalsExtra,
        regular: withdrawalsRegular,
      },
      startingBalanceOfSavingsPortfolio,
      savingsPortfolioStockAllocation,
      withdrawalFromSavingsRate,
      firstYearOfSomeRun,
      legacyByPercentile,
      endingBalanceOfSavingsPortfolioByPercentile,
    }

    return status.canceled ? null : result
  }

  terminate() {
    this._workers.forEach(x => x.terminate())
  }
}
const _mergeNumberByWorkerByYearsFromNowByRun = (
  numberByWorkerByYearsFromNowByRun: number[][][]
) => {
  const numOfRetirementYears = numberByWorkerByYearsFromNowByRun[0].length
  const numberByYearsFromNowByWorkerByRun = _.times(
    numOfRetirementYears,
    x => [] as number[][]
  )

  for (const numberByYearsFromNowByRun of numberByWorkerByYearsFromNowByRun) {
    numberByYearsFromNowByRun.forEach((x, i) =>
      numberByYearsFromNowByWorkerByRun[i].push(x)
    )
  }

  const numberByYearsFromNowByRun = numberByYearsFromNowByWorkerByRun.map(row =>
    _.flatten(row)
  )

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

  const legacyByRun = _.flatten(runsByWorker.map(s => s.legacyByRun))
  const endingBalanceOfSavingsPortfolioByRun = _.flatten(
    runsByWorker.map(s => s.endingBalanceOfSavingsPortfolioByRun)
  )
  return {
    byYearsFromNowByRun,
    firstYearOfSomeRun,
    legacyByRun,
    endingBalanceOfSavingsPortfolioByRun,
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
