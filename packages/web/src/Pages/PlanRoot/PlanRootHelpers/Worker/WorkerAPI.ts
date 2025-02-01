import { PlanParams } from '@tpaw/common'

export type WorkerArgs = {
  type: 'parseAndMigratePlanParams'
  taskID: string
  args: {
    isPreBase: true
    planParamsHistoryStr: readonly { id: string; params: string }[]
  }
}

export type WorkerResult =
  | { type: 'error'; message: string }
  | {
      type: 'parseAndMigratePlanParams'
      taskID: string
      result: { id: string; params: PlanParams }[]
    }
