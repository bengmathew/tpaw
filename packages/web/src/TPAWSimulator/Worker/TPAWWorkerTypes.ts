import {TPAWSimulationForYear} from '../RunTPAWSimulation'
import {TPAWParams} from '../TPAWParams'

export type TPAWWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {params: TPAWParams; numRuns: number}
    }
  | {type: 'sortRows'; taskID: string; args: {data: number[][]}}

export type TPAWWorkerResult =
  | {
      type: 'runSimulation'
      taskID: string
      result: {
        byYearsFromNowByRun: {
          withdrawals: {
            total: number[][]
            essential: number[][]
            extra: number[][]
            regular: number[][]
          }
          startingBalanceOfSavingsPortfolio: number[][]
          savingsPortfolioStockAllocation: number[][]
          withdrawalFromSavingsRate: number[][]
        }
        firstYearOfSomeRun: TPAWSimulationForYear
        legacyByRun: number[]
        endingBalanceOfSavingsPortfolioByRun: number[]
      }
    }
  | {type: 'sortRows'; taskID: string; result: number[][]}

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']
