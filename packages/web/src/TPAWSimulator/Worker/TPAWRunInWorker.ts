import _ from 'lodash'
import {fGet, noCase} from '../../Utils/Utils'
import {StatsTools} from '../StatsTools'
import {TPAWParams} from '../TPAWParams'
import {
  TPAWWorkerResult,
  TPAWWorkerRunSimulationResult,
} from './TPAWWorkerTypes'

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
  ) {
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
      withdrawalsByYearsIntoRetirementByRun,
      legacyByRun,
      firstYearOfSomeRun,
    } = _mergeRunsByWorker(runsByWorker)

    const withdrawalsByWorkerByYearsIntoRetirementByRunSorted =
      await Promise.all(
        this._workers.map((worker, i) => {
          const {start, length} = _loadBalance(i, numRuns, this._workers.length)
          return this._sortRows(
            worker,
            withdrawalsByYearsIntoRetirementByRun.slice(start, start + length)
          )
        })
      )

    if (status.canceled) return null
    const withdrawalsByYearsIntoRetirementByRunSorted = _.flatten(
      withdrawalsByWorkerByYearsIntoRetirementByRunSorted
    )
    const withdrawalsByYearsIntoRetirementByPercentile =
      StatsTools.pickPercentilesFromSorted(
        withdrawalsByYearsIntoRetirementByRunSorted,
        percentiles
      )
    const withdrawalsByPercentileByYearsIntoRetirement = StatsTools.pivot(
      withdrawalsByYearsIntoRetirementByPercentile
    ).map((data, i) => ({data, percentile: percentiles[i]}))

    const legacyByRunSorted = (
      await this._sortRows(this._workers[0], [legacyByRun])
    )[0]

    const legacyByPercentile = StatsTools.pickPercentilesFromSorted(
      [legacyByRunSorted],
      percentiles
    )[0].map((legacy, i) => ({legacy, percentile: percentiles[i]}))

    const maxWithdrawal = Math.max(
      ...withdrawalsByPercentileByYearsIntoRetirement.map(x =>
        Math.max(...x.data)
      )
    )
    const minWithdrawal = Math.min(
      ...withdrawalsByPercentileByYearsIntoRetirement.map(x =>
        Math.min(...x.data)
      )
    )
    const result = {
      withdrawalsByPercentileByYearsIntoRetirement,
      firstYearOfSomeRun,
      legacyByPercentile,
      minWithdrawal,
      maxWithdrawal,
    }

    return status.canceled ? null : result
  }

  terminate() {
    this._workers.forEach(x => x.terminate())
  }
}

const _mergeRunsByWorker = (
  runsByWorker: TPAWWorkerRunSimulationResult[]
): TPAWWorkerRunSimulationResult => {
  const withdrawalsByYearsIntoRetirementByWorkerByRun = _.range(
    runsByWorker[0].withdrawalsByYearsIntoRetirementByRun.length
  ).map(x => [] as number[][])

  for (const runs of runsByWorker) {
    runs.withdrawalsByYearsIntoRetirementByRun.forEach((x, i) =>
      withdrawalsByYearsIntoRetirementByWorkerByRun[i].push(x)
    )
  }
  const withdrawalsByYearsIntoRetirementByRun =
    withdrawalsByYearsIntoRetirementByWorkerByRun.map(row => _.flatten(row))
  const firstYearOfSomeRun = runsByWorker[0].firstYearOfSomeRun

  const legacyByRun = _.flatten(runsByWorker.map(s => s.legacyByRun))
  return {
    withdrawalsByYearsIntoRetirementByRun,
    firstYearOfSomeRun,
    legacyByRun,
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
