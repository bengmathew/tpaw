import _ from 'lodash'
import {noCase} from '../../Utils/Utils'
import {historicalReturns} from '../HistoricalReturns'
import {getNumYears} from '../TPAWParamsExt'
import {runSimulationInWASM} from './RunSimulationInWASM'
import {runSPAWSimulation} from './RunSPAWSimulation'
import {runTPAWSimulation} from './RunTPAWSimulation'
import {SimulationResult} from './SimulationResult'
import {TPAWWorkerArgs, TPAWWorkerResult} from './TPAWWorkerTypes'

const randomStore = new Map<number, Map<number, number>>()

const IS_JS = false

const _mapGetOrDefault = <K, V>(
  map: Map<K, V>,
  key: K,
  getDefault: () => V
) => {
  const result = map.get(key)
  if (result !== undefined) return result
  const d = getDefault()
  map.set(key, d)
  return d
}
const memoizedRandom = (runIndex: number, year: number) => {
  const valuesForRun = _mapGetOrDefault(randomStore, runIndex, () => new Map())
  return _mapGetOrDefault(valuesForRun, year, () =>
    _.random(historicalReturns.length - 1)
  )
}

addEventListener('message', async event => {
  const eventData: TPAWWorkerArgs = event.data
  const {taskID} = eventData
  switch (eventData.type) {
    case 'runSimulation':
      {
        const {result, perf} = IS_JS
          ? await doJS(eventData)
          : await runSimulationInWASM(
              eventData.args.params,
              eventData.args.numRuns
            )

        const reply: TPAWWorkerResult = {
          type: 'runSimulation',
          taskID,
          result: {
            ...result,
            perf: [
              ['runs', perf.runs],
              ['post', perf.post],
              ['rest', perf.rest],
              ['total', perf.total],
            ],
          },
        }

        const transferables: Transferable[] = IS_JS
          ? [
              ...result.byYearsFromNowByRun.savingsPortfolio.start.balance.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.withdrawals.essential.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.withdrawals.discretionary.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.withdrawals.regular.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.withdrawals.total.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.withdrawals.fromSavingsPortfolioRate.map(
                x => x.buffer
              ),
              ...result.byYearsFromNowByRun.savingsPortfolio.afterWithdrawals.allocation.stocks.map(
                x => x.buffer
              ),
              result.byRun.endingBalanceOfSavingsPortfolio.buffer,
            ]
          : [
              result.byYearsFromNowByRun.savingsPortfolio.start.balance[0]
                .buffer,
              result.byYearsFromNowByRun.savingsPortfolio.withdrawals
                .essential[0].buffer,
              result.byYearsFromNowByRun.savingsPortfolio.withdrawals
                .discretionary[0].buffer,
              result.byYearsFromNowByRun.savingsPortfolio.withdrawals.regular[0]
                .buffer,
              result.byYearsFromNowByRun.savingsPortfolio.withdrawals.total[0]
                .buffer,
              result.byYearsFromNowByRun.savingsPortfolio.withdrawals
                .fromSavingsPortfolioRate[0].buffer,
              result.byYearsFromNowByRun.savingsPortfolio.afterWithdrawals
                .allocation.stocks[0].buffer,
              result.byRun.endingBalanceOfSavingsPortfolio.buffer,
            ]
        ;(postMessage as any)(reply, transferables)
      }
      break
    case 'sortRows':
      {
        let start = performance.now()
        const {data} = eventData.args

        const wasm = await import('simulator')
        const sorted = IS_JS
          ? data.map(row => row.sort())
          : data.map(row => wasm.sort(row))

        const perf = performance.now() - start
        const reply: TPAWWorkerResult = {
          type: 'sortRows',
          taskID,
          result: {data: sorted, perf},
        }
        ;(postMessage as any)(
          reply,
          sorted.map(x => x.buffer)
        )
      }
      break
    default:
      noCase(eventData)
  }
})

async function doJS(
  eventData: Extract<TPAWWorkerArgs, {type: 'runSimulation'}>
) {
  const start0 = performance.now()

  const {numRuns, params} = eventData.args
  const numYears = getNumYears(params.original)

  const getArrays = () => _.times(numYears, () => new Float64Array(numRuns))

  let start = performance.now()
  let runs: SimulationResult[]
  switch (params.strategy) {
    case 'TPAW': {
      const resultsFromUsingExpectedReturns = runTPAWSimulation(params, {
        type: 'useExpectedReturns',
      })
      runs = _.range(numRuns).map((x, i) =>
        runTPAWSimulation(params, {
          type: 'useHistoricalReturns',
          resultsFromUsingExpectedReturns,
          randomIndexesIntoHistoricalReturnsByYear: year =>
            memoizedRandom(i, year),
        })
      )
      break
    }
    case 'SPAW': {
      const resultsFromUsingExpectedReturns = runSPAWSimulation(params, {
        type: 'useExpectedReturns',
      })
      runs = _.range(numRuns).map((x, i) =>
        runSPAWSimulation(params, {
          type: 'useHistoricalReturns',
          resultsFromUsingExpectedReturns,
          randomIndexesIntoHistoricalReturnsByYear: year =>
            memoizedRandom(i, year),
        })
      )
      break
    }
    default:
      noCase(params.strategy)
  }
  const perfRuns = performance.now() - start
  start = performance.now()

  const result = {
    byYearsFromNowByRun: {
      savingsPortfolio: {
        start: {
          balance: getArrays(),
        },
        withdrawals: {
          essential: getArrays(),
          discretionary: getArrays(),
          regular: getArrays(),
          total: getArrays(),
          fromSavingsPortfolioRate: getArrays(),
        },
        afterWithdrawals: {
          allocation: {
            stocks: getArrays(),
          },
        },
      },
    },
    byRun: {
      endingBalanceOfSavingsPortfolio: new Float64Array(
        runs.map(x => x.endingBalanceOfSavingsPortfolio)
      ),
    },
  }

  // Performance is better than looping for each of these individually.
  runs.forEach((run, r) =>
    run.byYearFromNow.forEach((year, y) => {
      const t = result.byYearsFromNowByRun
      t.savingsPortfolio.start.balance[y][r] =
        year.savingsPortfolio.start.balance
      t.savingsPortfolio.withdrawals.essential[y][r] =
        year.savingsPortfolio.withdrawals.essential
      t.savingsPortfolio.withdrawals.discretionary[y][r] =
        year.savingsPortfolio.withdrawals.discretionary
      t.savingsPortfolio.withdrawals.regular[y][r] =
        year.savingsPortfolio.withdrawals.regular
      t.savingsPortfolio.withdrawals.total[y][r] =
        year.savingsPortfolio.withdrawals.total
      t.savingsPortfolio.withdrawals.fromSavingsPortfolioRate[y][r] =
        year.savingsPortfolio.withdrawals.fromSavingsPortfolioRate
      t.savingsPortfolio.afterWithdrawals.allocation.stocks[y][r] =
        year.savingsPortfolio.afterWithdrawals.allocation.stocks
    })
  )

  const perfPost = performance.now() - start
  const perfTotal = performance.now() - start0
  const perfRest = perfTotal - perfRuns - perfPost
  return {
    result,
    perf: {runs: perfRuns, post: perfPost, rest: perfRest, total: perfTotal},
  }
}
