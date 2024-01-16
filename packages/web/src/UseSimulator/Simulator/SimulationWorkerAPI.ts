import { MarketData, PlanParams } from '@tpaw/common'
import { CurrentPortfolioBalance } from '../../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'
import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'

export type SimulationWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {
        params: PlanParamsProcessed
        runs: SimpleRange
        randomSeed: number
      }
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
  | {
      type: 'parseAndMigratePlanParams'
      taskID: string
      args: {
        isPreBase: true
        planParamsHistoryStr: readonly { id: string; params: string }[]
      }
    }
  | {
      type: 'estimateCurrentPortfolioBalance'
      taskID: string
      args: {
        planId: string
        isPreBase: boolean
        planParamsHistory: (
          | { cached: true; id: string }
          | { cached: false; id: string; params: PlanParams }
        )[]
        estimationTimestamp: number
        ianaTimezoneName: string
        marketData: MarketData.Data
      }
    }

export type SimulationWorkerResult =
  | { type: 'error'; message: string }
  | {
      type: 'runSimulation'
      taskID: string
      result: RunSimulationInWASMResult
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
        oneYear: _SampleReturnsStatsForWindowSize
        fiveYear: _SampleReturnsStatsForWindowSize
        tenYear: _SampleReturnsStatsForWindowSize
        thirtyYear: _SampleReturnsStatsForWindowSize
        perf: number
      }
    }
  | {
      type: 'parseAndMigratePlanParams'
      taskID: string
      result: { id: string; params: PlanParams }[]
    }
  | {
      type: 'estimateCurrentPortfolioBalance'
      taskID: string
      result: CurrentPortfolioBalance.ByMonthInfo
    }

export type SimulationWorkerSortResult = Extract<
  SimulationWorkerResult,
  { type: 'sort' }
>['result']

export type SimulationWorkerCalculateSampledAnnualReturn = Extract<
  SimulationWorkerResult,
  { type: 'getSampledReturnStats' }
>['result']

type _SampleReturnsStatsForWindowSize = {
  n: number
  mean: number
  ofLog: {
    mean: number
    variance: number
    standardDeviation: number
    n: number
  }
}
