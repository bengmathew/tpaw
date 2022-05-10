import {TPAWSimulationForYear} from '../RunTPAWSimulation'
import {TPAWParams} from '../TPAWParams'
import { TPAWParamsProcessed } from '../TPAWParamsProcessed'

export type TPAWWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {params: TPAWParamsProcessed; numRuns: number}
    }
  | {type: 'sortRows'; taskID: string; args: {data: Float64Array[]}}

export type TPAWWorkerResult =
  | {
      type: 'runSimulation'
      taskID: string
      result: {
        byYearsFromNowByRun: {
          withdrawals: {
            total: Float64Array[]
            essential: Float64Array[]
            extra: Float64Array[]
            regular: Float64Array[]
          }
          startingBalanceOfSavingsPortfolio: Float64Array[]
          savingsPortfolioStockAllocation: Float64Array[]
          withdrawalFromSavingsRate: Float64Array[]
        }
        firstYearOfSomeRun: TPAWSimulationForYear
        legacyByRun: Float64Array
        endingBalanceOfSavingsPortfolioByRun: Float64Array
        perf: [['runs', number], ['selectAndPivotPre', number], ['selectAndPivot', number], ['total', number]]
      }
    }
  | {type: 'sortRows'; taskID: string; result: Float64Array[]}

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']
