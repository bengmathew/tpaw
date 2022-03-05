import _ from 'lodash'
import {noCase} from '../../Utils/Utils'
import {historicalReturns} from '../HistoricalReturns'
import {runTPAWSimulation} from '../RunTPAWSimulation'
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
        const simulationUsingExpectedReturns = runTPAWSimulation(
          params,
          null
        ).byYearFromNow
        let start = performance.now()
        const runs = _.range(numRuns).map((x, i) =>
          runTPAWSimulation(params, {
            simulationUsingExpectedReturns,
            randomIndexesIntoHistoricalReturnsByYear: year =>
              memoizedRandom(i, year),
          })
        )
        const perfRuns = performance.now() - start
        start = performance.now()

        const numYears = runs[0].byYearFromNow.length
        const getArrays = () =>
          _.times(numYears, () => new Float64Array(numRuns))
        const result = {
          byYearsFromNowByRun: {
            withdrawals: {
              total: getArrays(),
              essential: getArrays(),
              extra: getArrays(),
              regular: getArrays(),
            },
            startingBalanceOfSavingsPortfolio: getArrays(),
            endingBalanceOfSavingsPortfolio: getArrays(),
            savingsPortfolioStockAllocation: getArrays(),
            withdrawalFromSavingsRate: getArrays(),
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
            t.withdrawals.total[y][r] = year.withdrawalAchieved.total
            t.withdrawals.essential[y][r] = year.withdrawalAchieved.essential
            t.withdrawals.extra[y][r] = year.withdrawalAchieved.extra
            t.withdrawals.regular[y][r] = year.withdrawalAchieved.regular
            t.startingBalanceOfSavingsPortfolio[y][r] =
              year.wealthAndSpending.startingBalanceOfSavingsPortfolio
            t.endingBalanceOfSavingsPortfolio[y][r] =
              year.savingsPortfolioEndingBalance
            t.savingsPortfolioStockAllocation[y][r] =
              year.savingsPortfolioAllocation.asPercentage.stocks ?? 0
            t.withdrawalFromSavingsRate[y][r] =
              year.withdrawalAchieved.fromSavingsRate
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
          ...result.byYearsFromNowByRun.withdrawals.total.map(x => x.buffer),
          ...result.byYearsFromNowByRun.withdrawals.essential.map(
            x => x.buffer
          ),
          ...result.byYearsFromNowByRun.withdrawals.extra.map(x => x.buffer),
          ...result.byYearsFromNowByRun.withdrawals.regular.map(x => x.buffer),
          ...result.byYearsFromNowByRun.startingBalanceOfSavingsPortfolio.map(
            x => x.buffer
          ),
          ...result.byYearsFromNowByRun.endingBalanceOfSavingsPortfolio.map(
            x => x.buffer
          ),
          ...result.byYearsFromNowByRun.savingsPortfolioStockAllocation.map(
            x => x.buffer
          ),
          ...result.byYearsFromNowByRun.withdrawalFromSavingsRate.map(
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
