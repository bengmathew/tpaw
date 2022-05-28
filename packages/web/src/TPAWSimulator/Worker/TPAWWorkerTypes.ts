import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'

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
          savingsPortfolio: {
            start: {balance: Float64Array[]}
            withdrawals: {
              essential: Float64Array[]
              discretionary: Float64Array[]
              regular: Float64Array[]
              total: Float64Array[]
              fromSavingsPortfolioRate: Float64Array[]
            }
            afterWithdrawals: {
              allocation: {stocks: Float64Array[]}
            }
          }
        }
        firstYearOfSomeRun: {savingsPortfolio: SavingsPortfolioThroughAYear.End}
        legacyByRun: Float64Array
        endingBalanceOfSavingsPortfolioByRun: Float64Array
        perf: [
          ['runs', number],
          ['selectAndPivotPre', number],
          ['selectAndPivot', number],
          ['total', number]
        ]
      }
    }
  | {type: 'sortRows'; taskID: string; result: Float64Array[]}

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']
