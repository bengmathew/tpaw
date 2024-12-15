import { MarketData, PlanParams } from '@tpaw/common'
import { PortfolioBalanceEstimation } from '../../Pages/PlanRoot/PlanRootHelpers/PortfolioBalanceEstimation'
import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import {
  PlanParamsRust,
  DataForMarketBasedPlanParamValues,
} from '@tpaw/simulator'

export type SimulationWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {
        currentPortfolioBalanceAmount: number
        planParamsRust: PlanParamsRust
        marketData: DataForMarketBasedPlanParamValues
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


export type SimulationWorkerSortResult = Extract<
  SimulationWorkerResult,
  { type: 'sort' }
>['result']
