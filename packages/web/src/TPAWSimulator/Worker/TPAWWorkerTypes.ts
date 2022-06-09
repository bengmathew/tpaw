import {TPAWParamsProcessed} from '../TPAWParamsProcessed'

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
        byRun: {
          endingBalanceOfSavingsPortfolio: Float64Array
        }
        perf: [
          ['runs', number],
          ['post', number],
          ['rest', number],
          ['total', number]
        ]
      }
    }
  | {
      type: 'sortRows'
      taskID: string
      result: {data: Float64Array[]; perf: number}
    }

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']

export type TPAWWorkerSortResult = Extract<
  TPAWWorkerResult,
  {type: 'sortRows'}
>['result']
