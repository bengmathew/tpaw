import { assertFalse } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { StatsTools } from '../../Utils/StatsTools'
import { assert, fGet, noCase } from '../../Utils/Utils'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import {
  FirstMonthSavingsPortfolioDetail,
  getFirstMonthSavingsPortfolioDetail,
} from './GetFirstMonthSavingsPortfolioDetail'
import { mergeSimulationWorkerRuns } from './MergeSimulationWorkerRuns'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import {
  SimulationWorkerArgs,
  SimulationWorkerResult,
  SimulationWorkerSortResult,
} from './SimulationWorkerAPI'

export type NumberArrByPercentileByMonthsFromNow = {
  byPercentileByMonthsFromNow: { data: number[]; percentile: number }[]
}
export type NumberArrWithIdByPercentileByMonthsFromNow = {
  id: string
  byPercentileByMonthsFromNow: { data: number[]; percentile: number }[]
}

// TODO:  Testing. Delete
// const MULTI_THREADED = true
const MULTI_THREADED = false

 const PERCENTILES_STR = ['5', '50', '95'] as const
 type Percentile = (typeof PERCENTILES_STR)[number]
 const PERCENTILES = PERCENTILES_STR.map((x) => parseInt(x))

export type SimulationResult = {
  args: SimulationArgs
  numSimulationsActual: number
  numRunsWithInsufficientFunds: number
  savingsPortfolio: {
    start: {
      balance: NumberArrByPercentileByMonthsFromNow
    }
    withdrawals: {
      essential: {
        total: NumberArrByPercentileByMonthsFromNow
        byId: NumberArrWithIdByPercentileByMonthsFromNow[]
      }
      discretionary: {
        total: NumberArrByPercentileByMonthsFromNow
        byId: NumberArrWithIdByPercentileByMonthsFromNow[]
      }
      regular: NumberArrByPercentileByMonthsFromNow
      total: NumberArrByPercentileByMonthsFromNow
      fromSavingsPortfolioRate: NumberArrByPercentileByMonthsFromNow
    }
    afterWithdrawals: {
      allocation: {
        stocks: NumberArrByPercentileByMonthsFromNow
      }
    }
  }
  totalPortfolio: {
    afterWithdrawals: {
      allocation: {
        stocks: NumberArrByPercentileByMonthsFromNow
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
    Record<
      'ofBase' | 'ofLog',
      {
        mean: number
        variance: number
        standardDeviation: number
        n: number
      }
    >
  >
  info: Rust.RunResultInfo
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

export type SimulationArgs = {
  currentPortfolioBalanceAmount: number
  planParamsRust: Rust.PlanParamsRust
  marketData: Rust.DataForMarketBasedPlanParamValues
  planParamsNorm: PlanParamsNormalized
  planParamsProcessed: PlanParamsProcessed
  numOfSimulationForMonteCarloSampling: number
  randomSeed: number
}

export class Simulator {
  private _workers: Worker[]
  private _resolvers = {
    runSimulation: new Map<
      string,
      (value: RunSimulationInWASMResult) => void
    >(),
    sort: new Map<string, (value: SimulationWorkerSortResult) => void>(),
  }
  constructor() {
    const numWorkers = MULTI_THREADED ? navigator.hardwareConcurrency || 4 : 1
    this._workers = _.range(numWorkers).map(
      () => new Worker(new URL('./SimulationWorker.ts', import.meta.url)),
    )

    this._workers.forEach((worker) => {
      worker.onmessage = (event) => {
        const data = event.data as SimulationWorkerResult
        if (data.type === 'error') {
          throw new Error(`Error in worker: ${data.message}`)
        }
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
          case 'parseAndMigratePlanParams':
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
    randomSeed: number,
    currentPortfolioBalanceAmount: number,
    planParamsRust: Rust.PlanParamsRust,
    marketData: Rust.DataForMarketBasedPlanParamValues,
    planParamsNorm: PlanParamsNormalized,
    planParamsProcessed: PlanParamsProcessed,
  ): Promise<RunSimulationInWASMResult> {
    const taskID = _.uniqueId()
    const message: Extract<SimulationWorkerArgs, { type: 'runSimulation' }> = {
      taskID,
      type: 'runSimulation',
      args: {
        planParamsRust,
        marketData,
        currentPortfolioBalanceAmount,
        runs,
        planParamsNorm,
        planParamsProcessed,
        randomSeed,
      },
    }
    worker.postMessage(message)
    return new Promise<RunSimulationInWASMResult>((resolve) =>
      this._resolvers.runSimulation.set(taskID, resolve),
    )
  }
  private _sort(
    worker: Worker,
    data: Float64Array[],
  ): Promise<SimulationWorkerSortResult> {
    const taskID = _.uniqueId()
    const transferables: Transferable[] = data.map((x) => x.buffer)
    const message: Extract<SimulationWorkerArgs, { type: 'sort' }> = {
      taskID,
      type: 'sort',
      args: { data },
    }
    worker.postMessage(message, transferables)
    return new Promise<SimulationWorkerSortResult>((resolve) =>
      this._resolvers.sort.set(taskID, resolve),
    )
  }

  async runSimulations(
    status: { canceled: boolean },
    args: SimulationArgs,
  ): Promise<SimulationResult | null> {
    const {
      currentPortfolioBalanceAmount,
      planParamsNorm,
      planParamsRust,
      marketData,
      planParamsProcessed,
      numOfSimulationForMonteCarloSampling,
      randomSeed,
    } = args
    const start0 = performance.now()
    let start = performance.now()
    const numSimulationsActual = _getNumSimulationsActual(
      planParamsProcessed,
      planParamsNorm,
      numOfSimulationForMonteCarloSampling,
    )
    const percentiles = PERCENTILES

    const runsByWorker = await Promise.all(
      // Note, make sure each worker always gets a consistent run range so that
      // the memoizedRandom will not miss just due to different run range.
      this._workers.map((worker, i) =>
        this._runSimulation(
          worker,
          _loadBalance(i, numSimulationsActual, this._workers.length),
          randomSeed,
          currentPortfolioBalanceAmount,
          planParamsRust,
          marketData,
          planParamsNorm,
          planParamsProcessed,
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
    } = mergeSimulationWorkerRuns(runsByWorker)

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

    const info = runsByWorker[0].info
    const firstMonthOfSomeRun = getFirstMonthSavingsPortfolioDetail(
      runsByWorker[0].byMonthsFromNowByRun.savingsPortfolio,
      planParamsProcessed,
    )

    const withdrawalsEssentialById = _separateExtraWithdrawals(
      planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending.essential,
      withdrawalsEssential,
    )

    const withdrawlsDiscretionaryById = _separateExtraWithdrawals(
      planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending
        .discretionary,
      withdrawalsDiscretionary,
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

    const result: SimulationResult = {
      args,
      numSimulationsActual,
      numRunsWithInsufficientFunds,
      savingsPortfolio: {
        start: {
          balance: startingBalanceOfSavingsPortfolio,
        },
        withdrawals: {
          essential: {
            total: withdrawalsEssential,
            byId: withdrawalsEssentialById,
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
      info,
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

export const _separateExtraWithdrawals = (
  processedEntries: PlanParamsProcessed['byMonth']['adjustmentsToSpending']['extraSpending']['essential'],
  combinedExtraWithdrawals: Omit<NumberArrByPercentileByMonthsFromNow, 'id'>,
): NumberArrWithIdByPercentileByMonthsFromNow[] =>
  processedEntries.byId.map((processedEntry) => ({
    id: processedEntry.id,
    ..._mapByPercentileByMonthsFromNow(
      combinedExtraWithdrawals,
      (value, monthsFromNow) => {
        if (
          !SimpleRange.Closed.isIn(monthsFromNow, processedEntry.validRange)
        ) {
          return 0
        }
        const currMonthParams = processedEntries.total[monthsFromNow]
        const withdrawalTargetForThisMonth = fGet(processedEntry.values)[
          monthsFromNow
        ]
        if (withdrawalTargetForThisMonth === 0) return 0
        const ratio = withdrawalTargetForThisMonth / currMonthParams
        assert(!isNaN(ratio)) // withdrawalTargetForThisMonth ?>0 imples denominator is not 0.
        return value * ratio
      },
    ),
  }))

const _mapByPercentileByMonthsFromNow = (
  x: NumberArrByPercentileByMonthsFromNow,
  fn: (x: number, i: number) => number,
): NumberArrByPercentileByMonthsFromNow => {
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
  planParamsNorm: PlanParamsNormalized,
  numOfSimulationForMonteCarloSampling: number,
) => {
  switch (planParamsNorm.advanced.sampling.type) {
    case 'monteCarlo':
      return numOfSimulationForMonteCarloSampling
    case 'historical': {
      return (
        planParamsProcessed.historicalReturnsAdjusted.stocks.stats.log.n -
        planParamsNorm.ages.simulationMonths.numMonths +
        1
      )
    }
    default:
      noCase(planParamsNorm.advanced.sampling)
  }
}
