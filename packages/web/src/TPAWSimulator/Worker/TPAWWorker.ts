import _ from 'lodash'
import { noCase } from '../../Utils/Utils'
import { historicalReturns } from '../HistoricalReturns'
import { runTPAWSimulation } from '../RunTPAWSimulation'
import { StatsTools } from '../StatsTools'
import { TPAWWorkerArgs, TPAWWorkerResult } from './TPAWWorkerTypes'

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
        const runs = _.range(numRuns).map((x, i) =>
          runTPAWSimulation(params, {
            randomIndexesIntoHistoricalReturnsByYear: year =>
              memoizedRandom(i, year),
          })
        )

        const withdrawalsByRunByYearFromNow = runs.map(x =>
          x.byYearFromNow.map(x => x.withdrawal)
        )
        const withdrawalsByYearFromNowByRun = StatsTools.pivot(
          withdrawalsByRunByYearFromNow
        )
        const withdrawalsByYearsIntoRetirementByRun =
          withdrawalsByYearFromNowByRun.slice(
            params.age.retirement - params.age.start
          )

        const legacyByRun = runs.map(x => x.legacy)

        const result = {
          withdrawalsByYearsIntoRetirementByRun,
          firstYearOfSomeRun: runs[0].byYearFromNow[0],
          legacyByRun,
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
