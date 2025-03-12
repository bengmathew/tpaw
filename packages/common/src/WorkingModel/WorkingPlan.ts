import cloneJSON from 'fast-json-clone'
import _ from 'lodash'
import { CalendarDayFns } from '../Misc/CalendarDayFns'
import {
  getFullDatedDefaultPlanParams,
  getFullDatelessDefaultPlanParams,
} from '../Params/PlanParams/DefaultPlanParams'
import { normalizePlanParams } from '../Params/PlanParams/NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from '../Params/PlanParams/NormalizePlanParams/NormalizePlanParamsInverse'
import { planParamsGuard } from '../Params/PlanParams/PlanParams'
import { getPlanParamsChangeActionImpl } from '../Params/PlanParams/PlanParamsChangeAction/GetPlanParamsChangeActionImpl/GetPlanParamsChangeActionImpl'
import { RPC } from '../RPC/RPC'
import { assert, assertFalse, block, fGet, letIn, noCase } from '../Utils'
import { PlanParamsHistoryStoreFns } from './PlanParamsHistoryStoreFns'
import { ReverseHeadIndex } from '../Misc/ReverseHeadIndex'

export namespace WorkingPlan {
  export type PlanParamsHistoryItem = PlanParamsHistoryStoreFns.Item & {
    committed: boolean
  }

  export type Model = {
    location: 'server' | 'files' | 'links'
    userId: string
    planId: string
    label: string
    resetCount: number
    addedToServerAt: number
    sortTime: number
    lastSyncAt: number
    reverseHeadIndex: number
    planParamsHistoryPostBase: PlanParamsHistoryItem[]
  }

  export type PlanNonUpdateAction = Extract<
    RPC.PlanTransaction,
    { type: 'update' }
  >

  // TODO: Move to web.
  export namespace Client {
    export type PlanActionAndTarget =
      | {
          targetPlanId: string
          action:
            | Exclude<
                RPC.PlanTransaction,
                { type: 'update' } | { type: 'create' } | { type: 'copy' }
              >
            | RPC.PlanUpdateAction
            | {
                type: 'copy'
                destPlanId: string
                destLocation: Model['location']
              }
        }
      | { action: Extract<RPC.PlanTransaction, { type: 'create' }> }

    export const applyPlanActionAndTarget = (
      planActionAndTarget: PlanActionAndTarget,
      models: Model[],
      timestampForNorm: number,
      timestampForChange: number,
      ianaTimezoneName: string,
      mutableTransactionsByPlanId: Map<string, RPC.PlanTransaction[]>, // Note, this is mutated.
    ): Model[] => {
      if (!('targetPlanId' in planActionAndTarget)) {
        switch (planActionAndTarget.action.type) {
          case 'create':
            // TODO:
            assertFalse()
          default:
            noCase(planActionAndTarget.action.type)
        }
      } else {
        const { currModel, transactions, otherModels } = block(() => {
          const planId = planActionAndTarget.targetPlanId
          const currModel = fGet(models.find((m) => m.planId === planId))
          const transactions =
            currModel.location === 'server'
              ? mutableTransactionsByPlanId.get(planId) || []
              : null
          if (transactions && !mutableTransactionsByPlanId.has(planId)) {
            mutableTransactionsByPlanId.set(planId, transactions)
          }
          return {
            currModel,
            transactions,
            otherModels: models.filter((x) => x !== currModel),
          }
        })

        switch (planActionAndTarget.action.type) {
          case 'delete': {
            transactions?.push({ type: 'delete' })
            return otherModels
          }
          case 'copy': {
            transactions?.push({
              type: 'copy',
              destPlanId: planActionAndTarget.action.destPlanId,
            })
            const now = Date.now()
            const model = {
              location: planActionAndTarget.action.destLocation,
              userId: currModel.userId,
              planId: planActionAndTarget.action.destPlanId,
              label: `Copy of ${currModel.label}`,
              resetCount: 0,
              addedToServerAt: now,
              sortTime: now,
              lastSyncAt: now,
              reverseHeadIndex: currModel.reverseHeadIndex,
              planParamsHistoryPostBase:
                currModel.planParamsHistoryPostBase.map((x) => ({
                  id: x.id,
                  change: x.change,
                  planParamsUnmigrated: x.planParamsUnmigrated,
                  planParams: x.planParams,
                  committed: false,
                })),
            }
            return [model, ...otherModels]
          }
          case 'reset': {
            const { changeId } = planActionAndTarget.action
            transactions?.push({ type: 'reset', changeId })
            const { isDated } = fGet(currModel.planParamsHistoryPostBase[0])
              .planParams.datingInfo
            const startingPlanParams = isDated
              ? getFullDatedDefaultPlanParams(Date.now(), ianaTimezoneName)
              : getFullDatelessDefaultPlanParams(Date.now())

            const now = Date.now()
            const model: Model = {
              location: currModel.location,
              userId: currModel.userId,
              planId: currModel.planId,
              label: currModel.label,
              resetCount: currModel.resetCount + 1,
              addedToServerAt: currModel.addedToServerAt,
              sortTime: now,
              lastSyncAt: now,
              reverseHeadIndex: 0,
              planParamsHistoryPostBase: [
                {
                  id: changeId,
                  change: { type: 'start', value: null },
                  planParams: startingPlanParams,
                  planParamsUnmigrated: startingPlanParams,
                  committed: false,
                },
              ],
            }
            return [model, ...otherModels]
          }
          case 'planParamChange':
          case 'setHead':
          case 'setLabel':
            const model = _applyUpdateAction(
              planActionAndTarget.action,
              currModel,
              timestampForNorm,
              timestampForChange,
              ianaTimezoneName,
              transactions,
            )
            return [model, ...otherModels]
          default:
            noCase(planActionAndTarget.action)
        }
      }
    }

    const _applyUpdateAction = (
      action: RPC.PlanUpdateAction,
      model: Model,
      timestampForNorm: number,
      timestampForChange: number,
      ianaTimezoneName: string,
      mutableTransactions: RPC.PlanTransaction[] | null, // Note, this is mutated.
    ): Model => {
      const transaction = mutableTransactions
        ? block(() => {
            const lastTransaction = _.last(mutableTransactions)
            if (lastTransaction?.type === 'update') {
              return lastTransaction
            } else {
              const result = {
                type: 'update' as const,
                currentResetCount: model.resetCount,
                actions: [],
              }
              mutableTransactions?.push(result)
              return result
            }
          })
        : null

      const helperResult = _applyUpdateActionHelper(
        action,
        model,
        timestampForNorm,
        timestampForChange,
        ianaTimezoneName,
        { allowFailure: false, allowMerge: true },
      )

      switch (helperResult.type) {
        case 'success':
          if (!helperResult.isNoOp) transaction?.actions.push(action)
          return model
        // TODO: Test
        case 'merge': {
          // Note. In principle it is possible that different non planParamChange
          // action might have happened in between, but it is extremely unlikely
          // since it would have to happen through user input in the UI, in
          // sub-second time. We assume this cannot happen. If this assumption
          // fails in future (eg. faster UI through keyboard shortcuts), this
          // can be relaxed but applying the merge only if these conditions hold,
          // rather than asserting.
          if (transaction) {
            const lastAction = _.last(transaction.actions)
            assert(lastAction?.type === 'planParamChange')
            assert(lastAction.changeId === helperResult.lastId)
            const lastHistoryItem = _.last(model.planParamsHistoryPostBase)
            assert(lastHistoryItem?.id === lastAction.changeId)
          }
          // There cannot be a redo action because that would mean an undo
          // happened in between, which we assume cannot happen as per above note,
          // so head must be pointing to the end of history.
          assert(model.reverseHeadIndex === 0)

          // Undo the last action.
          transaction?.actions.pop()
          const modelAfterUndo = {
            ...model,
            planParamsHistoryPostBase: model.planParamsHistoryPostBase.slice(
              0,
              -1,
            ),
          }
          return _applyUpdateAction(
            action,
            modelAfterUndo,
            timestampForNorm,
            timestampForChange,
            ianaTimezoneName,
            mutableTransactions,
          )
        }
        case 'failed':
          assertFalse()
        default:
          noCase(helperResult)
      }
    }
  }

  // TODO: Move to server.
  export namespace Server {
    export const applyUpdateAction = (
      action: RPC.PlanUpdateAction,
      model: Model,
      ianaTimezoneName: string,
    ): { isSuccess: true; model: Model } | { isSuccess: false } => {
      const now = Date.now()
      const result = _applyUpdateActionHelper(
        action,
        model,
        now,
        now,
        ianaTimezoneName,
        {
          allowFailure: true,
          allowMerge: false,
        },
      )
      switch (result.type) {
        case 'success':
          return { isSuccess: true, model: result.model }
        case 'failed':
          return { isSuccess: false }
        case 'merge':
          assert(false)
      }
    }
  }

  const _applyUpdateActionHelper = (
    action: RPC.PlanUpdateAction,
    model: Model,
    timestampForNorm: number,
    timestampForChange: number,
    ianaTimezoneName: string,
    {
      allowFailure,
      allowMerge,
    }: { allowFailure: boolean; allowMerge: boolean },
  ):
    | { type: 'success'; model: Model; isNoOp: boolean }
    | { type: 'failed' }
    | { type: 'merge'; lastId: string } => {
    const getFailed = () =>
      allowFailure ? { type: 'failed' as const } : assertFalse()
    const getSuccess = (model: Model, isNoOp: 'noOp' | 'notNoOp') =>
      ({ type: 'success', model, isNoOp: isNoOp === 'noOp' }) as const

    switch (action.type) {
      case 'setLabel': {
        const trimLabel = action.label.trim()
        const now = Date.now()
        return model.label === trimLabel
          ? getSuccess(model, 'noOp')
          : getSuccess(
              {
                location: model.location,
                userId: model.userId,
                planId: model.planId,
                label: trimLabel,
                resetCount: model.resetCount,
                addedToServerAt: model.addedToServerAt,
                sortTime: model.sortTime,
                lastSyncAt: now,
                reverseHeadIndex: model.reverseHeadIndex,
                planParamsHistoryPostBase: model.planParamsHistoryPostBase,
              },
              'notNoOp',
            )
      }
      case 'setHead': {
        if (
          ReverseHeadIndex.fGet(
            model.reverseHeadIndex,
            model.planParamsHistoryPostBase,
          ).id === action.targetId
        ) {
          return getSuccess(model, 'noOp')
        }
        const headIndex = model.planParamsHistoryPostBase.findIndex(
          (v) => v.id === action.targetId,
        )
        if (headIndex === -1) {
          return getFailed()
        }
        const now = Date.now()
        return getSuccess(
          {
            location: model.location,
            userId: model.userId,
            planId: model.planId,
            label: model.label,
            resetCount: model.resetCount,
            addedToServerAt: model.addedToServerAt,
            sortTime: now,
            lastSyncAt: now,
            reverseHeadIndex: ReverseHeadIndex.fromHeadIndex(
              headIndex,
              model.planParamsHistoryPostBase.length,
            ),
            planParamsHistoryPostBase: model.planParamsHistoryPostBase,
          },
          'notNoOp',
        )
      }
      case 'planParamChange': {
        return _applyPlanParamsChangeAction(
          action,
          model,
          timestampForNorm,
          timestampForChange,
          ianaTimezoneName,
          { allowMerge, allowFailure },
        )
      }
      default:
        noCase(action)
    }
  }

  const _applyPlanParamsChangeAction = (
    {
      changeId,
      planParamsChangeAction,
    }: Extract<RPC.PlanUpdateAction, { type: 'planParamChange' }>,
    model: Model,
    timestampForNorm: number,
    timestampForChange: number,
    ianaTimezoneName: string,
    {
      allowMerge,
      // TODO: implement
      allowFailure,
    }: { allowMerge: boolean; allowFailure: boolean },
  ):
    | { type: 'success'; model: Model; isNoOp: boolean }
    | { type: 'failed' }
    | { type: 'merge'; lastId: string } => {
    const planParamsHistoryUpToHead = ReverseHeadIndex.sliceToInclusive(
      model.reverseHeadIndex,
      model.planParamsHistoryPostBase,
    )
    const currentHistoryItem = fGet(_.last(planParamsHistoryUpToHead))
    assert(timestampForChange > currentHistoryItem.planParams.timestamp)

    // clone() because values from the action might end up in the planParams
    // and we don't want that link.
    const { applyToClone, merge } = getPlanParamsChangeActionImpl(
      _.clone(planParamsChangeAction),
    )

    const planParamsNorm = normalizePlanParams(currentHistoryItem.planParams, {
      timestamp: timestampForNorm,
      calendarDay: CalendarDayFns.fromTimestamp(
        timestampForNorm,
        ianaTimezoneName,
      ),
    })
    const planParamsFromNorm = normalizePlanParamsInverse(planParamsNorm)
    const nextPlanParams = letIn(
      cloneJSON(planParamsFromNorm),
      (clone) => applyToClone(clone, planParamsNorm) ?? clone,
    )

    const isNoOp =
      // setCurrentPortfolio balance is a special case, because it is is
      // sensitive to timestamps (if the plan is dated). So even if the value
      // is the same but timestamp is different we need to update it.
      nextPlanParams.datingInfo.isDated &&
      planParamsChangeAction.type === 'setCurrentPortfolioBalance'
        ? false
        : _.isEqual(nextPlanParams, planParamsFromNorm)
    if (isNoOp) return { type: 'success', model, isNoOp: true }

    nextPlanParams.timestamp = timestampForChange
    if (planParamsChangeAction.type !== 'setCurrentPortfolioBalance') {
      // Sanity check, because we use planParamsFromNorm.timestamp below.
      assert(
        planParamsFromNorm.timestamp ===
          currentHistoryItem.planParams.timestamp,
      )
      nextPlanParams.wealth.portfolioBalance =
        planParamsFromNorm.wealth.portfolioBalance.isDatedPlan &&
        planParamsFromNorm.wealth.portfolioBalance.updatedHere
          ? {
              isDatedPlan: true,
              updatedHere: false,
              updatedAtId: currentHistoryItem.id,
              updatedTo: planParamsFromNorm.wealth.portfolioBalance.amount,
              updatedAtTimestamp: planParamsFromNorm.timestamp,
            }
          : _.cloneDeep(planParamsFromNorm.wealth.portfolioBalance)
    }
    nextPlanParams.results = null
    planParamsGuard(nextPlanParams).force()

    const planParamsHistoryPostBase: PlanParamsHistoryItem[] = [
      ...planParamsHistoryUpToHead,
      {
        id: changeId,
        change: planParamsChangeAction,
        planParamsUnmigrated: nextPlanParams,
        planParams: nextPlanParams,
        committed: false,
      },
    ]

    if (
      allowMerge &&
      currentHistoryItem.change.type === planParamsChangeAction.type &&
      nextPlanParams.timestamp - planParamsFromNorm.timestamp < 1000 &&
      (typeof merge === 'function' ? merge(currentHistoryItem.change) : merge)
    ) {
      return { type: 'merge', lastId: currentHistoryItem.id }
    }

    return {
      type: 'success',
      model: {
        location: model.location,
        userId: model.userId,
        planId: model.planId,
        label: model.label,
        resetCount: model.resetCount,
        addedToServerAt: model.addedToServerAt,
        sortTime: timestampForChange,
        lastSyncAt: timestampForChange,
        reverseHeadIndex: 0,
        planParamsHistoryPostBase,
      },
      isNoOp: false,
    }
  }
}
