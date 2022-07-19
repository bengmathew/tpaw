import {SimpleRange} from '../../Utils/SimpleRange'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'

export type TPAWWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {params: TPAWParamsProcessed; runs: SimpleRange}
    }
  | {type: 'sort'; taskID: string; args: {data: Float64Array[]}}
  | {type: 'calculateOneOverCV'; taskID: string; args: {data: Float64Array[]}}
  | {type: 'clearMemoizedRandom'; taskID: string}

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
            excessWithdrawals: {
              regular: Float64Array[]
            }
            afterWithdrawals: {
              allocation: {stocks: Float64Array[]}
            }
          }
          totalPortfolio: {
            afterWithdrawals: {
              allocation: {stocks: Float64Array[]}
            }
          }
        }
        byRun: {
          numInsufficientFundYears: Int32Array
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
      type: 'sort'
      taskID: string
      result: {data: Float64Array[]; perf: number}
    }
  | {
      type: 'calculateOneOverCV'
      taskID: string
      result: {data: Float64Array; perf: number}
    }
  | {
      type: 'clearMemoizedRandom'
      taskID: string
    }

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']

export type TPAWWorkerSortResult = Extract<
  TPAWWorkerResult,
  {type: 'sort'}
>['result']

export type TPAWWorkerCalculateOneOverCVResult = Extract<
  TPAWWorkerResult,
  {type: 'calculateOneOverCV'}
>['result']
