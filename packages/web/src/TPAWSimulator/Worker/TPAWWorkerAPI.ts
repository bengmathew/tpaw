import { MarketData, PlanParams } from '@tpaw/common'
import { CurrentPortfolioBalance } from '../../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'
import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { RunSimulationInWASMResult } from './RunSimulationInWASMResult'
import { TPAWWorkerResultTypes } from './TPAWWorkerResultTypes'

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
  | { type: 'clearMemoizedRandom'; taskID: string }

export type TPAWWorkerResult =
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
      result: TPAWWorkerResultTypes.GetSampledResultStats
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
  | {
      type: 'clearMemoizedRandom'
      taskID: string
    }

export type TPAWWorkerSortResult = Extract<
  TPAWWorkerResult,
  { type: 'sort' }
>['result']

export type TPAWWorkerCalculateSampledAnnualReturn = Extract<
  TPAWWorkerResult,
  { type: 'getSampledReturnStats' }
>['result']
