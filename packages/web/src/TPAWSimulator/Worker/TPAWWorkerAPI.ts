import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'

export type TPAWWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: { params: PlanParamsProcessed; runs: SimpleRange }
    }
  | { type: 'sort'; taskID: string; args: { data: Float64Array[] } }
  | {
      type: 'getSampledReturnStats'
      taskID: string
      args: {
        monthlyReturns: number[]
        blockSize: number
        numMonths: number
      }
    }
  | { type: 'clearMemoizedRandom'; taskID: string }

export type TPAWWorkerResult =
  | {
      type: 'runSimulation'
      taskID: string
      result: {
        byMonthsFromNowByRun: {
          savingsPortfolio: {
            start: { balance: Float64Array[] }
            withdrawals: {
              essential: Float64Array[]
              discretionary: Float64Array[]
              regular: Float64Array[]
              total: Float64Array[]
              fromSavingsPortfolioRate: Float64Array[]
            }
            afterWithdrawals: {
              allocation: { stocks: Float64Array[] }
            }
          }
          totalPortfolio: {
            afterWithdrawals: {
              allocation: { stocks: Float64Array[] }
            }
          }
        }
        byRun: {
          numInsufficientFundMonths: Int32Array
          endingBalanceOfSavingsPortfolio: Float64Array
        }
        averageAnnualReturns: {
          stocks: number
          bonds: number
        }
        perf: [
          ['runs', number],
          ['post', number],
          ['rest', number],
          ['total', number],
        ]
      }
    }
  | {
      type: 'sort'
      taskID: string
      result: { data: Float64Array[]; perf: number }
    }
  | {
      type: 'getSampledReturnStats'
      taskID: string
      result: {
        oneYear:SampleReturnsStatsForWindowSize
        fiveYear:SampleReturnsStatsForWindowSize
        tenYear:SampleReturnsStatsForWindowSize
        thirtyYear:SampleReturnsStatsForWindowSize
        perf: number
      }
    }
  | {
      type: 'clearMemoizedRandom'
      taskID: string
    }

type SampleReturnsStatsForWindowSize = {
  n: number
  mean: number
  ofLog: {
    mean: number
    variance: number
    standardDeviation: number
    n: number
  }
}

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  { type: 'runSimulation' }
>['result']

export type TPAWWorkerSortResult = Extract<
  TPAWWorkerResult,
  { type: 'sort' }
>['result']

export type TPAWWorkerCalculateSampledAnnualReturn = Extract<
  TPAWWorkerResult,
  { type: 'getSampledReturnStats' }
>['result']
