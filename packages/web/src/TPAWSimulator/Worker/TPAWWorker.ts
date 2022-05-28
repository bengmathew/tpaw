import _ from 'lodash'
import {noCase} from '../../Utils/Utils'
import {historicalReturns} from '../HistoricalReturns'
import {extendTPAWParams} from '../TPAWParamsExt'
import {runSPAWSimulation} from './RunSPAWSimulation'
import {runTPAWSimulation} from './RunTPAWSimulation'
import {SimulationResult} from './SimulationResult'
import {TPAWWorkerArgs, TPAWWorkerResult} from './TPAWWorkerTypes'

const randomStore = new Map<number, Map<number, number>>()

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

addEventListener('message', event => {
  const {taskID, ...eventData}: TPAWWorkerArgs = event.data
  switch (eventData.type) {
    case 'runSimulation':
      {
        let start0 = performance.now()
        const {numRuns, params} = eventData.args
        const paramsExt = extendTPAWParams(params.original)

        let runs: SimulationResult[]

        let start = performance.now()

        switch (params.strategy) {
          case 'TPAW': {
            const resultsFromUsingExpectedReturns = runTPAWSimulation(
              params,
              paramsExt,
              {type: 'useExpectedReturns'}
            )
            runs = _.range(numRuns).map((x, i) =>
              runTPAWSimulation(params, paramsExt, {
                type: 'useHistoricalReturns',
                resultsFromUsingExpectedReturns,
                randomIndexesIntoHistoricalReturnsByYear: year =>
                  memoizedRandom(i, year),
              })
            )
            break
          }
          case 'SPAW': {
            const resultsFromUsingExpectedReturns = runSPAWSimulation(
              params,
              paramsExt,
              {type: 'useExpectedReturns'}
            )
            runs = _.range(numRuns).map((x, i) =>
              runSPAWSimulation(params, paramsExt, {
                type: 'useHistoricalReturns',
                resultsFromUsingExpectedReturns,
                randomIndexesIntoHistoricalReturnsByYear: year =>
                  memoizedRandom(i, year),
              })
            )
            break
            break
          }
          default:
            noCase(params.strategy)
        }
        const perfRuns = performance.now() - start
        start = performance.now()

        const numYears = runs[0].byYearFromNow.length
        const getArrays = () =>
          _.times(numYears, () => new Float64Array(numRuns))
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
          firstYearOfSomeRun: runs[0].byYearFromNow[0],
          legacyByRun: new Float64Array(runs.map(x => x.legacy)),
          endingBalanceOfSavingsPortfolioByRun: new Float64Array(
            runs.map(x => x.endingBalanceOfSavingsPortfolio)
          ),
        }

        const perfSelectAndPivotPre = performance.now() - start
        start = performance.now()
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

        const perfSelectAndPivot = performance.now() - start
        start = performance.now()

        const reply: TPAWWorkerResult = {
          type: 'runSimulation',
          taskID,
          result: {
            ...result,
            perf: [
              ['runs', perfRuns],
              ['selectAndPivotPre', perfSelectAndPivotPre],
              ['selectAndPivot', perfSelectAndPivot],
              ['total', performance.now() - start0],
            ],
          },
        }
        ;(postMessage as any)(reply, [
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
          result.legacyByRun.buffer,
          result.endingBalanceOfSavingsPortfolioByRun.buffer,
        ])
      }
      break
    case 'sortRows':
      {
        const {data} = eventData.args
        const result = data.map(row => row.sort())
        const reply: TPAWWorkerResult = {
          type: 'sortRows',
          taskID,
          result,
        }
        ;(postMessage as any)(
          reply,
          result.map(x => x.buffer)
        )
      }
      break
    default:
      noCase(eventData)
  }
})
