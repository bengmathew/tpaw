import _ from 'lodash'
import {noCase} from '../../Utils/Utils'
import {historicalReturns} from '../HistoricalReturns'
import {runTPAWSimulation, TPAWSimulationForYear} from '../RunTPAWSimulation'
import {StatsTools} from '../StatsTools'
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
        const {numRuns, params} = eventData.args
        const simulationUsingExpectedReturns = runTPAWSimulation(
          params,
          null
        ).byYearFromNow
        const runs = _.range(numRuns).map((x, i) =>
          runTPAWSimulation(params, {
            simulationUsingExpectedReturns,
            randomIndexesIntoHistoricalReturnsByYear: year =>
              memoizedRandom(i, year),
          })
        )

        const _selectAndPivot = (fn: (x: TPAWSimulationForYear) => number) =>
          StatsTools.pivot(runs.map(x => x.byYearFromNow.map(fn)))

        const result = {
          byYearsFromNowByRun: {
            withdrawals: {
              total: _selectAndPivot(x => x.withdrawalAchieved.total),
              essential: _selectAndPivot(x => x.withdrawalAchieved.essential),
              extra: _selectAndPivot(x => x.withdrawalAchieved.extra),
              regular: _selectAndPivot(x => x.withdrawalAchieved.regular),
            },
            startingBalanceOfSavingsPortfolio: _selectAndPivot(
              x => x.wealthAndSpending.startingBalanceOfSavingsPortfolio
            ),
            endingBalanceOfSavingsPortfolio: _selectAndPivot(
              x => x.savingsPortfolioEndingBalance
            ),
            savingsPortfolioStockAllocation: _selectAndPivot(
              x => x.savingsPortfolioAllocation.asPercentage.stocks ?? 0
            ),
            withdrawalFromSavingsRate: _selectAndPivot(
              x => x.withdrawalAchieved.fromSavingsRate
            ),
          },
          firstYearOfSomeRun: runs[0].byYearFromNow[0],
          legacyByRun: runs.map(x => x.legacy),
          endingBalanceOfSavingsPortfolioByRun: runs.map(
            x => x.endingBalanceOfSavingsPortfolio
          ),
        }

        const reply: TPAWWorkerResult = {
          type: 'runSimulation',
          taskID,
          result,
        }
        ;(postMessage as any)(reply)
      }
      break
    case 'sortRows':
      {
        const {data} = eventData.args
        const result = data.map(row => _.sortBy(row))
        const reply: TPAWWorkerResult = {type: 'sortRows', taskID, result}
        ;(postMessage as any)(reply)
      }
      break
    default:
      noCase(eventData)
  }
})
