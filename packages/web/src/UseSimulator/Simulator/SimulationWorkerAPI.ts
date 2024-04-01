import { MarketData, PlanParams } from '@tpaw/common'
import { CurrentPortfolioBalance } from '../../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'
import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'

export type SimulationWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {
        currentPortfolioBalanceAmount: number
        planParamsNorm: PlanParamsNormalized
        planParamsProcessed: PlanParamsProcessed
        runs: SimpleRange
        randomSeed: number
      }
    }
  | { type: 'sort'; taskID: string; args: { data: Float64Array[] } }
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
