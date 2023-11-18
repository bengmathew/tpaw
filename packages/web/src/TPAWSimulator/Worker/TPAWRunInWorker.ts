import { assertFalse, NonPlanParams, ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { StatsTools } from '../../Utils/StatsTools'
import { assert, fGet, noCase } from '../../Utils/Utils'
import { PlanParamsExtended } from '../ExtentPlanParams'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import {
  firstMonthSavingsPortfolioDetail,
  FirstMonthSavingsPortfolioDetail,
} from './FirstMonthSavingsPortfolioDetail'
import { mergeWorkerRuns } from './MergeWorkerRuns'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import {
  TPAWWorkerArgs,
  TPAWWorkerCalculateSampledAnnualReturn,
  TPAWWorkerResult,
  TPAWWorkerSortResult,
} from './TPAWWorkerAPI'

export type TPAWRunInWorkerByPercentileByMonthsFromNow = {
  byPercentileByMonthsFromNow: { data: number[]; percentile: number }[]
}

const MULTI_THREADED = true
export const PERCENTILES_STR = ['5', '50', '95'] as const
export type Percentile = (typeof PERCENTILES_STR)[number]
export const PERCENTILES = PERCENTILES_STR.map((x) => parseInt(x))

export type TPAWRunInWorkerResult = {
  numSimulationsActual: number
  numRunsWithInsufficientFunds: number
  savingsPortfolio: {
    start: {
      balance: TPAWRunInWorkerByPercentileByMonthsFromNow
    }
    withdrawals: {
      essential: {
        total: TPAWRunInWorkerByPercentileByMonthsFromNow
        byId: Map<string, TPAWRunInWorkerByPercentileByMonthsFromNow>
      }
      discretionary: {
        total: TPAWRunInWorkerByPercentileByMonthsFromNow
        byId: Map<string, TPAWRunInWorkerByPercentileByMonthsFromNow>
      }
      regular: TPAWRunInWorkerByPercentileByMonthsFromNow
      total: TPAWRunInWorkerByPercentileByMonthsFromNow
      fromSavingsPortfolioRate: TPAWRunInWorkerByPercentileByMonthsFromNow
    }
    afterWithdrawals: {
      allocation: {
        stocks: TPAWRunInWorkerByPercentileByMonthsFromNow
      }
    }
  }
  totalPortfolio: {
    afterWithdrawals: {
      allocation: {
        stocks: TPAWRunInWorkerByPercentileByMonthsFromNow
      }
    }
  }
  endingBalanceOfSavingsPortfolioByPercentile: {
    data: number
    percentile: number
  }[]
  firstMonthOfSomeRun: FirstMonthSavingsPortfolioDetail
  annualStatsForSampledReturns: Record<
    'stocks' | 'bonds',
    { n: number } & Record<
      'ofBase' | 'ofLog',
      {
        mean: number
        variance: number
        standardDeviation: number
        n: number
      }
    >
  >
  perf: {
    main: [
      ['num-workers', number],
      ['simulation', number],
      ['merge-simulation', number],
      ['sort-and-pick-percentiles', number],
      ['post', number],
      ['rest', number],
      ['total', number],
    ]
    sortAndPickPercentilesYearly: [
      ['pre', number],
      ['slowest-sort-worker', number],
      ['sort', number],
      ['post', number],
      ['percentile', number],
      ['rest', number],
      ['total', number],
    ]
    slowestSimulationWorker: [
      ['runs', number],
      ['post', number],
      ['rest', number],
      ['total', number],
    ]
  }
}

export class TPAWRunInWorker {
  private _workers: Worker[]
  private _resolvers = {
    runSimulation: new Map<
      string,
      (value: RunSimulationInWASMResult) => void
    >(),
    sort: new Map<string, (value: TPAWWorkerSortResult) => void>(),
    getSampledReturnStats: new Map<
      string,
      (value: TPAWWorkerCalculateSampledAnnualReturn) => void
    >(),
    clearMemoizedRandom: new Map<string, () => void>(),
  }
  constructor() {
    const numWorkers = MULTI_THREADED ? navigator.hardwareConcurrency || 4 : 1
    this._workers = _.range(numWorkers).map(
      () => new Worker(new URL('./TPAWWorker.ts', import.meta.url)),
    )

    this._workers.forEach((worker) => {
      worker.onerror = (event) => {
        console.dir(event)
      }
      worker.addEventListener('unhandledrejection', (event) => {
        console.dir(event)
      })
      worker.onmessage = (event) => {
        const data = event.data as TPAWWorkerResult
        const { taskID } = data
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
          case 'getSampledReturnStats': {
            const resolve = fGet(
              this._resolvers.getSampledReturnStats.get(taskID),
            )
            this._resolvers.getSampledReturnStats.delete(taskID)
            resolve(data.result)
            break
          }
          case 'clearMemoizedRandom': {
            const resolve = fGet(
              this._resolvers.clearMemoizedRandom.get(taskID),
            )
            this._resolvers.clearMemoizedRandom.delete(taskID)
            resolve()
            break
          }
          case 'parseAndMigratePlanParams':
          case 'estimateCurrentPortfolioBalance':
            assertFalse()
          default:
            noCase(data)
        }
      }
    })
  }

  private _runSimulation(
    worker: Worker,
    runs: SimpleRange,
    params: PlanParamsProcessed,
  ): Promise<RunSimulationInWASMResult> {
    const taskID = _.uniqueId()
    const message: Extract<TPAWWorkerArgs, { type: 'runSimulation' }> = {
      taskID,
      type: 'runSimulation',
      args: { runs, params },
    }
    worker.postMessage(message)
    return new Promise<RunSimulationInWASMResult>((resolve) =>
      this._resolvers.runSimulation.set(taskID, resolve),
    )
  }
  private _sort(
    worker: Worker,
    data: Float64Array[],
  ): Promise<TPAWWorkerSortResult> {
    const taskID = _.uniqueId()
    const transferables: Transferable[] = data.map((x) => x.buffer)
    const message: Extract<TPAWWorkerArgs, { type: 'sort' }> = {
      taskID,
      type: 'sort',
      args: { data },
    }
    worker.postMessage(message, transferables)
    return new Promise<TPAWWorkerSortResult>((resolve) =>
      this._resolvers.sort.set(taskID, resolve),
    )
  }

  private _getSampledReturnStats(
    worker: Worker,
    monthlyReturns: number[],
    blockSize: number,
    numMonths: number,
  ): Promise<TPAWWorkerCalculateSampledAnnualReturn> {
    const taskID = _.uniqueId()
    const message: Extract<TPAWWorkerArgs, { type: 'getSampledReturnStats' }> =
      {
        taskID,
        type: 'getSampledReturnStats',
        args: { monthlyReturns, blockSize, numMonths },
      }
    worker.postMessage(message)
    return new Promise<TPAWWorkerCalculateSampledAnnualReturn>((resolve) =>
      this._resolvers.getSampledReturnStats.set(taskID, resolve),
    )
  }

  private _clearMemoizedRandom(worker: Worker): Promise<void> {
    const taskID = _.uniqueId()
    const message: Extract<TPAWWorkerArgs, { type: 'clearMemoizedRandom' }> = {
      type: 'clearMemoizedRandom',
      taskID,
    }
    worker.postMessage(message)
    return new Promise<void>((resolve) =>
      this._resolvers.clearMemoizedRandom.set(taskID, resolve),
    )
  }

  async clearMemoizedRandom(): Promise<void> {
    await Promise.all(
      this._workers.map((worker) => this._clearMemoizedRandom(worker)),
    )
  }

  async getSampledReturnStats(
    monthlyReturns: number[],
    blockSize: number,
    numMonths: number,
  ) {
    const byWorker = await Promise.all(
      this._workers.map((worker, i) =>
        this._getSampledReturnStats(
          worker,
          monthlyReturns,
          blockSize,
          _loadBalance(i, numMonths, this._workers.length).length,
        ),
      ),
    )

    const merge = (yearsByWorker: (typeof byWorker)[0]['oneYear'][]) => {
      const totalN = _.sumBy(yearsByWorker, (x) => x.n)
      return {
        n: totalN,
        mean: _.sumBy(yearsByWorker, (x) => x.mean * x.n) / totalN,
        ofLog: {
          mean: _.sumBy(yearsByWorker, (x) => x.ofLog.mean * x.n) / totalN,
          varianceAveragedOverThread:
            _.sumBy(yearsByWorker, (x) => x.ofLog.variance) / byWorker.length,
        },
      }
    }

    return {
      oneYear: merge(byWorker.map((x) => x.oneYear)),
      fiveYear: merge(byWorker.map((x) => x.fiveYear)),
      tenYear: merge(byWorker.map((x) => x.tenYear)),
      thirtyYear: merge(byWorker.map((x) => x.thirtyYear)),
    }
  }

  // async hasLiquidityProblem(
  //   numRuns: number,
  //   params: PlanParamsProcessed
  // ): Promise<boolean> {

  //   numRuns = _numRuns(planParamsExt, numRuns)
  //   const runsByWorker = await Promise.all(
  //     this._workers.map((worker, i) =>
  //       this._runSimulation(
  //         worker,
  //         _loadBalance(i, numRuns, this._workers.length),
  //         params
  //       )
  //     )
  //   )

  // }

  async runSimulations(
    status: { canceled: boolean },
    params: PlanParamsProcessed,
    planParamsExt: PlanParamsExtended,
    nonPlanParams: NonPlanParams,
  ): Promise<TPAWRunInWorkerResult | null> {
    const start0 = performance.now()
    let start = performance.now()
    const numSimulationsActual = _getNumSimulationsActual(
      params,
      planParamsExt,
      nonPlanParams,
    )
    const percentiles = PERCENTILES

    const runsByWorker = await Promise.all(
      this._workers.map((worker, i) =>
        this._runSimulation(
          worker,
          _loadBalance(i, numSimulationsActual, this._workers.length),
          params,
        ),
      ),
    )

    const perfSimulation = performance.now() - start
    start = performance.now()

    if (status.canceled) return null
    const {
      byMonthsFromNowByRun,
      byRun,
      annualStatsForSampledReturns,
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
        totalPortfolioStockAllocation,
      ],
      perf: perfSortAndPickPercentilesYearly,
    } = await this._sortAndPickPercentilesForByMonthsFromNowByRun(
      [
        byMonthsFromNowByRun.savingsPortfolio.start.balance,
        byMonthsFromNowByRun.savingsPortfolio.withdrawals.essential,
        byMonthsFromNowByRun.savingsPortfolio.withdrawals.discretionary,
        byMonthsFromNowByRun.savingsPortfolio.withdrawals.regular,
        byMonthsFromNowByRun.savingsPortfolio.withdrawals.total,
        byMonthsFromNowByRun.savingsPortfolio.withdrawals
          .fromSavingsPortfolioRate,
        byMonthsFromNowByRun.savingsPortfolio.afterWithdrawals.allocation
          .stocks,
        byMonthsFromNowByRun.totalPortfolio.afterWithdrawals.allocation.stocks,
      ],
      percentiles,
    )

    if (status.canceled) return null

    const endingBalanceOfSavingsPortfolioByPercentile =
      _sortAndPickPercentilesForByRun(
        byRun.endingBalanceOfSavingsPortfolio,
        percentiles,
      )
    if (status.canceled) return null

    const perfSortAndPickPercentiles = performance.now() - start
    start = performance.now()

    const firstMonthOfSomeRun = firstMonthSavingsPortfolioDetail(
      runsByWorker[0].byMonthsFromNowByRun.savingsPortfolio,
      params,
    )
    const withdrawlsEssentialById = new Map(
      _.values(
        planParamsExt.planParams.adjustmentsToSpending.extraSpending.essential,
      ).map((x) => [
        x.id,
        _separateExtraWithdrawal(
          x,
          params,
          planParamsExt,
          withdrawalsEssential,
          'essential',
        ),
      ]),
    )
    const withdrawlsDiscretionaryById = new Map(
      _.values(
        planParamsExt.planParams.adjustmentsToSpending.extraSpending
          .discretionary,
      ).map((x) => [
        x.id,
        _separateExtraWithdrawal(
          x,
          params,
          planParamsExt,
          withdrawalsDiscretionary,
          'discretionary',
        ),
      ]),
    )

    const numRunsWithInsufficientFunds = byRun.numInsufficientFundMonths.filter(
      (x) => x > 0,
    ).length

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
      numSimulationsActual,
      numRunsWithInsufficientFunds,
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
          fromSavingsPortfolioRate: _mapByPercentileByMonthsFromNow(
            withdrawalFromSavingsRate,
            (x) => Math.max(0, x),
          ),
        },
        afterWithdrawals: {
          allocation: {
            stocks: savingsPortfolioStockAllocation,
          },
        },
      },
      totalPortfolio: {
        afterWithdrawals: {
          allocation: {
            stocks: totalPortfolioStockAllocation,
          },
        },
      },
      firstMonthOfSomeRun,
      endingBalanceOfSavingsPortfolioByPercentile,
      annualStatsForSampledReturns,
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

  private _sortAndPickPercentilesForByMonthsFromNowByRun = async (
    numberByTypeByMonthsFromNowByRun: Float64Array[][],
    percentiles: number[],
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
    const numMonthsArr = _.uniq(
      numberByTypeByMonthsFromNowByRun.map((x) => x.length),
    )
    assert(numMonthsArr.length === 1)
    const numMonths = numMonthsArr[0]
    const numberByMonthsFromNowByRun = _.flatten(
      numberByTypeByMonthsFromNowByRun,
    )
    perf.pre = performance.now() - start
    start = performance.now()
    const numberByWorkerByMonthsFromNowByRunSorted = await Promise.all(
      this._workers.map((worker, i) => {
        const { start, length } = _loadBalance(
          i,
          numberByMonthsFromNowByRun.length,
          this._workers.length,
        )
        return this._sort(
          worker,
          numberByMonthsFromNowByRun.slice(start, start + length),
        )
      }),
    )
    perf.slowestSortWorker = Math.max(
      ...numberByWorkerByMonthsFromNowByRunSorted.map((x) => x.perf),
    )
    perf.sort = performance.now() - start
    start = performance.now()

    const numberByMonthsFromNowByRunSorted = _.flatten(
      numberByWorkerByMonthsFromNowByRunSorted.map((x) => x.data),
    )

    perf.post = performance.now() - start
    start = performance.now()

    const numberByTypeByMonthsFromNowByRunSorted =
      numberByTypeByMonthsFromNowByRun.map((x, i) =>
        numberByMonthsFromNowByRunSorted.slice(
          i * numMonths,
          (i + 1) * numMonths,
        ),
      )
    const result = numberByTypeByMonthsFromNowByRunSorted.map(
      (numberByMonthsFromNowByRunSorted) => {
        const numberByMonthsFromNowByPercentile =
          StatsTools.pickPercentilesFromSorted(
            numberByMonthsFromNowByRunSorted,
            percentiles,
          )

        const byPercentileByMonthsFromNow = StatsTools.pivot(
          numberByMonthsFromNowByPercentile,
        ).map((data, i) => ({ data, percentile: percentiles[i] }))

        return { byPercentileByMonthsFromNow }
      },
    )

    perf.percentile = performance.now() - start
    perf.total = performance.now() - start0
    perf.rest =
      perf.total - (perf.pre + perf.sort + perf.post + perf.percentile)

    return { result, perf }
  }

  terminate() {
    this._workers.forEach((x) => x.terminate())
  }
}

const _sortAndPickPercentilesForByRun = (
  numberByRun: Float64Array,
  percentiles: number[],
) => {
  numberByRun.sort()
  return StatsTools.pickPercentilesFromSorted(
    [numberByRun],
    percentiles,
  )[0].map((data, i) => ({ data, percentile: percentiles[i] }))
}

const _loadBalance = (worker: number, numJobs: number, numWorkers: number) => {
  const remainder = numJobs % numWorkers
  const minJobsPerWorker = Math.floor(numJobs / numWorkers)
  const start = worker * minJobsPerWorker + Math.min(remainder, worker)
  const length = minJobsPerWorker + (worker < remainder ? 1 : 0)
  const end = start + length
  return { start, end, length }
}

const _separateExtraWithdrawal = (
  valueForMonthRange: ValueForMonthRange,
  params: PlanParamsProcessed,
  planParamsExt: PlanParamsExtended,
  x: TPAWRunInWorkerByPercentileByMonthsFromNow,
  type: 'discretionary' | 'essential',
): TPAWRunInWorkerByPercentileByMonthsFromNow => {
  const monthRange = planParamsExt.asMFN(valueForMonthRange.monthRange)

  return _mapByPercentileByMonthsFromNow(x, (value, monthsFromNow) => {
    if (monthsFromNow < monthRange.start || monthsFromNow > monthRange.end) {
      return 0
    }
    const currMonthParams =
      params.byMonth.adjustmentsToSpending.extraSpending[type].total[
        monthsFromNow
      ]
    const withdrawalTargetForThisMonth = fGet(
      params.byMonth.adjustmentsToSpending.extraSpending[type].byId[
        valueForMonthRange.id
      ].values,
    )[monthsFromNow]
    if (withdrawalTargetForThisMonth === 0) return 0
    const ratio = withdrawalTargetForThisMonth / currMonthParams
    assert(!isNaN(ratio)) // withdrawalTargetForThisMonth ?>0 imples denominator is not 0.
    return value * ratio
  })
}

const _mapByPercentileByMonthsFromNow = (
  x: TPAWRunInWorkerByPercentileByMonthsFromNow,
  fn: (x: number, i: number) => number,
): TPAWRunInWorkerByPercentileByMonthsFromNow => {
  const byPercentileByMonthsFromNow = x.byPercentileByMonthsFromNow.map(
    (x) => ({
      data: x.data.map(fn),
      percentile: x.percentile,
    }),
  )
  return { byPercentileByMonthsFromNow }
}

const _getNumSimulationsActual = (
  planParamsProcessed: PlanParamsProcessed,
  planParamsExt: PlanParamsExtended,
  nonPlanParams: NonPlanParams,
) => {
  const { planParams, numMonths } = planParamsExt
  switch (planParams.advanced.sampling.type) {
    case 'monteCarlo':
      return nonPlanParams.numOfSimulationForMonteCarloSampling
    case 'historical': {
      return (
        planParamsProcessed.historicalReturnsAdjusted.monthly.stocks.length -
        numMonths +
        1
      )
    }
    default:
      noCase(planParams.advanced.sampling.type)
  }
}
