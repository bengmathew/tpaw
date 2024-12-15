import {
  PlanParams,
  PlanParamsChangeAction,
  PlanParamsChangeActionCurrent,
  PlanPaths,
  SomePlanParams,
  assert,
  fGet,
  letIn,
  planParamsGuard,
} from '@tpaw/common'
import cloneJSON from 'fast-json-clone'
import _ from 'lodash'
import { useCallback, useMemo, useState } from 'react'
import * as uuid from 'uuid'
import { normalizePlanParams } from '../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from '../../../Simulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { CalendarDayFns } from '../../../Utils/CalendarDayFns'
import { useAssertConst } from '../../../Utils/UseAssertConst'
import { Config } from '../../Config'
import { getPlanParamsChangeActionImpl } from './GetPlanParamsChangeActionImpl/GetPlanParamsChangeActionImpl'
import { CurrentTimeInfo } from './UseCurrentTime'
import { useMarketData } from './WithMarketData'
import { useIANATimezoneName } from './WithNonPlanParams'
import { useWASM } from './WithWASM'

export type PlanParamsHistoryItem = {
  readonly id: string
  readonly change: PlanParamsChangeAction
  readonly paramsUnmigrated: SomePlanParams | null
  readonly params: PlanParams
}
export type WorkingPlanSrc = {
  readonly planId: string
  readonly planParamsPostBase: readonly PlanParamsHistoryItem[]
  readonly reverseHeadIndex: number
}

// Note 1. you can only grown this, not shrink it, because we assume
// reverseHeadIndex is always within the UNDO_DEPTH and reverseHeadIndex is
// persisted on the server.
// Note 2. This is the visible undo depth. The number of changes items to
// support it is  TARGET_UNDO_DEPTH + 1
export const TARGET_UNDO_DEPTH = Config.client.isProduction ? 100 : 10
export const REBASE_BUFFER = Config.client.isProduction ? 50 : 5

export type WorkingPlanInfo = ReturnType<typeof useWorkingPlan>

export const useWorkingPlan = (
  currentTimeInfo: CurrentTimeInfo,
  src: WorkingPlanSrc,
  planPaths: PlanPaths,
) => {
  const { ianaTimezoneName } = useIANATimezoneName()
  const { marketData } = useMarketData()
  const { wasm } = useWASM()

  const [workingPlan, setWorkingPlan] = useState(src)
  assert(workingPlan.reverseHeadIndex <= TARGET_UNDO_DEPTH)
  const headIndex = useMemo(
    () =>
      workingPlan.planParamsPostBase.length - 1 - workingPlan.reverseHeadIndex,
    [workingPlan],
  )

  const { planParams, planParamsUndoRedoStack } = useMemo(() => {
    const startingParams = workingPlan.planParamsPostBase[0].params
    assert(
      !startingParams.wealth.portfolioBalance.isDatedPlan ||
        startingParams.wealth.portfolioBalance.updatedHere,
    )
    const undos = workingPlan.planParamsPostBase.slice(0, headIndex + 1)
    const redos = workingPlan.planParamsPostBase.slice(headIndex + 1)
    const planParams = fGet(_.last(undos)).params
    return {
      planParams,
      planParamsUndoRedoStack: {
        undos,
        redos,
      },
    }
  }, [headIndex, workingPlan])

  const updatePlanParams = useCallback(
    <T extends PlanParamsChangeActionCurrent>(
      type: T['type'],
      value: T['value'],
    ) => {
      const currHistoryItem = fGet(_.last(planParamsUndoRedoStack.undos))
      assert(planParams === currHistoryItem.params)

      const change = { type, value } as T
      const { applyToClone, merge } = getPlanParamsChangeActionImpl(
        _.clone(change),
      )

      const planParamsNorm = normalizePlanParams(planParams, {
        timestamp: currentTimeInfo.currentTimestamp,
        calendarDay: CalendarDayFns.fromTimestamp(
          currentTimeInfo.currentTimestamp,
          ianaTimezoneName,
        ),
      })

      const planParamsFromNorm = normalizePlanParamsInverse(planParamsNorm)
      const nextPlanParams = letIn(
        cloneJSON(planParamsFromNorm),
        (clone) => applyToClone(clone, planParamsNorm) ?? clone,
      )
      if (_.isEqual(nextPlanParams, planParamsFromNorm)) {
        if (!nextPlanParams.datingInfo.isDated) return
        // setCurrentPortfolio balance is a special case, because it is is
        // sensitive to timestamps (if the plan is dated). So even if the value
        // is the same but timestamp is different we need to update it.
        if (change.type !== 'setCurrentPortfolioBalance') return
      }

      nextPlanParams.timestamp = currentTimeInfo.forceUpdateCurrentTime()
      if (change.type !== 'setCurrentPortfolioBalance') {
        nextPlanParams.wealth.portfolioBalance =
          planParamsFromNorm.wealth.portfolioBalance.isDatedPlan &&
          planParamsFromNorm.wealth.portfolioBalance.updatedHere
            ? {
                isDatedPlan: true,
                updatedHere: false,
                updatedAtId: currHistoryItem.id,
                updatedTo: planParamsFromNorm.wealth.portfolioBalance.amount,
                updatedAtTimestamp: planParamsFromNorm.timestamp,
              }
            : _.cloneDeep(planParamsFromNorm.wealth.portfolioBalance)
      }

      const planParamsPostBase = [
        ...planParamsUndoRedoStack.undos,
        {
          id: uuid.v4(),
          paramsUnmigrated: null,
          params: nextPlanParams,
          change,
        },
      ]
      if (
        currHistoryItem.change.type === change.type &&
        nextPlanParams.timestamp - planParamsFromNorm.timestamp < 1000 &&
        (typeof merge === 'function' ? merge(currHistoryItem.change) : merge)
      )
        planParamsPostBase.splice(-2, 1)

      nextPlanParams.results = null

      planParamsGuard(nextPlanParams).force()

      setWorkingPlan({
        planId: workingPlan.planId,
        planParamsPostBase,
        reverseHeadIndex: 0,
      })
    },
    [
      currentTimeInfo,
      ianaTimezoneName,
      planParams,
      planParamsUndoRedoStack.undos,
      workingPlan.planId,
    ],
  )
  useAssertConst([planPaths])

  const setPlanParamsHeadIndex = useCallback(
    (headIndex: number) => {
      const reverseHeadIndex =
        workingPlan.planParamsPostBase.length - 1 - headIndex

      if (reverseHeadIndex === workingPlan.reverseHeadIndex) return
      assert(reverseHeadIndex <= workingPlan.planParamsPostBase.length - 1)
      setWorkingPlan({
        planId: workingPlan.planId,
        planParamsPostBase: workingPlan.planParamsPostBase,
        reverseHeadIndex,
      })
    },
    [workingPlan],
  )

  const rebase = useMemo(() => {
    const planParamsPostBase = workingPlan.planParamsPostBase

    const rebaseIndex = _.findLastIndex(
      planParamsPostBase,
      (x, i) =>
        // not current index
        i > 0 &&
        // portfolio balance was updated here
        (!x.params.wealth.portfolioBalance.isDatedPlan ||
          x.params.wealth.portfolioBalance.updatedHere) &&
        // postBase size (include rebaseIndex) is more than size require for
        // undo + buffer.
        planParamsPostBase.length - i > TARGET_UNDO_DEPTH + 1 + REBASE_BUFFER,
    )

    if (rebaseIndex < 0) return null
    return ({ hard }: { hard: boolean }) => {
      const cutAndBase = planParamsPostBase.slice(0, rebaseIndex + 1)
      const base = planParamsPostBase[rebaseIndex]

      const newPlanParamsPostBase = [
        hard
          ? {
              ...base,
              change: { type: 'startCutByClient' as const, value: null },
            }
          : base,
        ...planParamsPostBase.slice(rebaseIndex + 1),
      ]
      setWorkingPlan((prev) => ({
        ...prev,
        planParamsPostBase: newPlanParamsPostBase,
      }))
      return cutAndBase
    }
  }, [workingPlan.planParamsPostBase])

  return {
    workingPlan,
    planParamsUndoRedoStack,
    updatePlanParams,
    setPlanParamsHeadIndex,
    rebase,
  }
}
