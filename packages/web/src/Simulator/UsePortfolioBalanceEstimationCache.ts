import {
  CalendarDay,
  MarketData,
  PlanParams,
  PlanParamsHistoryFns,
  assert,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { useCallback, useRef } from 'react'
import { normalizePlanParams } from './NormalizePlanParams/NormalizePlanParams'
import { getPlanParamsServer } from './SimulateOnServer/GetPlanParamsServer'
import {
  WirePortfolioBalanceEstimationArgs,
  WirePortfolioBalanceEstimationArgsNonMarketActionType,
} from './SimulateOnServer/Wire/wire_estimate_portfolio_balance_api'
import { WireSimulationArgs } from './SimulateOnServer/Wire/wire_simulate_api'
import { PortfolioBalanceEstimation } from '../Pages/PlanRoot/PlanRootHelpers/PortfolioBalanceEstimation'
import { fWASM } from './Simulator/GetWASM'
import { useMarketData } from '../Pages/PlanRoot/PlanRootHelpers/WithMarketData'
import { useIANATimezoneName } from '../Pages/PlanRoot/PlanRootHelpers/WithNonPlanParams'

export type PortfolioBalanceEstimationByDated =
  | {
      isDatedPlan: true
      detail: PortfolioBalanceEstimation.Detail
      currentBalance: number
    }
  | { isDatedPlan: false; currentBalance: number }

export type PortfolioBalanceEstimationCacheHandler = {
  wirePortfolioBalanceEstimationArgs: Exclude<
    WireSimulationArgs['currentPortfolioBalance'],
    undefined
  >
  handleEstimation: (
    estimation: PortfolioBalanceEstimation.Detail | null,
  ) => PortfolioBalanceEstimationByDated
}

export const getPortfolioBalanceEstimationCacheHandlerForDatelessPlan = (
  amount: number,
): PortfolioBalanceEstimationCacheHandler => ({
  wirePortfolioBalanceEstimationArgs: {
    $case: 'noEstimate',
    noEstimate: amount,
  },
  handleEstimation: (estimation) => {
    assert(!estimation)
    return { isDatedPlan: false, currentBalance: amount }
  },
})

export const usePortfolioBalanceEstimationCache = (
  planId: string,
  runTests: boolean,
) => {
  // TODO: Testing. Remove
  const { marketData } = useMarketData()
  const { ianaTimezoneName } = useIANATimezoneName()

  const cacheRef = useRef<{
    planId: string
    detail: PortfolioBalanceEstimation.Detail
  } | null>(null)

  const applyCache = useCallback(
    (
      planParamsHistory: readonly { id: string; params: PlanParams }[],
      simulationTimestamp: number,
      getCalendarDayFromTimestamp: (timestamp: number) => CalendarDay,
    ): PortfolioBalanceEstimationCacheHandler => {
      const cache = cacheRef.current
      const lastParams = fGet(_.last(planParamsHistory)).params
      if (!lastParams.wealth.portfolioBalance.isDatedPlan) {
        const amount = lastParams.wealth.portfolioBalance.amount
        return getPortfolioBalanceEstimationCacheHandlerForDatelessPlan(amount)
      }

      const firstParams = fGet(_.first(planParamsHistory)).params
      const firstPortfolioUpdate = firstParams.wealth.portfolioBalance
      assert(
        firstPortfolioUpdate.isDatedPlan && firstPortfolioUpdate.updatedHere,
      )

      const getResult = (
        beforeCut: PortfolioBalanceEstimation.Detail | null,
        portfolioBalanceAtCut: number,
        planParamsHistoryAfterCut: readonly {
          id: string
          params: PlanParams
        }[],
      ): PortfolioBalanceEstimationCacheHandler => ({
        wirePortfolioBalanceEstimationArgs: {
          $case: 'estimate',
          estimate: fGet(
            _getServerArgs(
              planParamsHistoryAfterCut,
              portfolioBalanceAtCut,
              simulationTimestamp,
              getCalendarDayFromTimestamp,
            ),
          ),
        },
        handleEstimation: (
          estimation: PortfolioBalanceEstimation.Detail | null,
        ) => {
          const detail = _joinDetails(beforeCut, fGet(estimation))

          // // TODO: Testing. Remove
          if (runTests) {
            _checkAgainstLocal(
              detail,
              planId,
              simulationTimestamp,
              ianaTimezoneName,
              marketData,
              planParamsHistory,
            )
          }
          cacheRef.current = { detail, planId }

          return {
            isDatedPlan: true,
            detail: {
              ...detail,
              actions: _mergeConsecutivePlanChanges(detail.actions),
            },
            currentBalance:
              _.last(detail.actions)?.stateChange.end.estimate ??
              detail.startState.estimate,
          }
        },
      })

      const getNoCacheResult = (reason: string) => {
        return getResult(null, firstPortfolioUpdate.amount, planParamsHistory)
      }
      if (!cache) return getNoCacheResult('starting')
      if (planId !== cache.planId) return getNoCacheResult('planChanged')
      // This catches if history is fetched.
      if (firstParams.timestamp !== cache.detail.startTimestamp)
        return getNoCacheResult('history')

      // Step 1: Cut cache to simulation timestamp.
      const cacheCutToSimulationTimestamp = _cutDetail(
        simulationTimestamp,
        cache.detail,
      )
      const planParamsHistoryTimestamps = new Set(
        planParamsHistory.map((x) => x.params.timestamp),
      )

      // Step 2: Cut cache to the last known plan change.
      const cacheCutToSimulationTimestampAndLastKnownPlanChange = block(() => {
        const lastKnownPlanChangeActionIndex =
          cacheCutToSimulationTimestamp.actions.findLastIndex(
            (x) =>
              x.args.type === 'planChange' &&
              planParamsHistoryTimestamps.has(x.timestamp),
          )
        if (lastKnownPlanChangeActionIndex === -1) return null

        // We can optimize further by not cutting so aggressively (some
        // non-planChange actions can be kept), but reasoning about correctness
        // is harder and there is not much benefit.
        const actions = cacheCutToSimulationTimestamp.actions.slice(
          0,
          lastKnownPlanChangeActionIndex + 1,
        )

        return {
          startTimestamp: cacheCutToSimulationTimestamp.startTimestamp,
          startState: cacheCutToSimulationTimestamp.startState,
          actions,
          endTimestamp: fGet(_.last(actions)).timestamp,
        }
      })
      if (!cacheCutToSimulationTimestampAndLastKnownPlanChange)
        return getNoCacheResult('four')

      // TODO: Testing. This catches if history is rebased. This is not strictly
      // necessary. Needed only for testing.
      // {
      //   if (
      //     cacheCutToSimulationTimestampAndLastKnownPlanChange.actions
      //       .filter((x) => x.args.type === 'planChange')
      //       .map((x) => x.timestamp)
      //       .some((x) => !planParamsHistoryTimestamps.has(x))
      //   ) {
      //     return getNoCacheResult('rebase')
      //   }
      // }

      const lastKnownPlanChangeAction = fGet(
        _.last(cacheCutToSimulationTimestampAndLastKnownPlanChange.actions),
      )

      return getResult(
        cacheCutToSimulationTimestampAndLastKnownPlanChange,
        // end.estimate and not start.estimate because the first item in history
        // is the starting state, and this is the value to use for that start
        // state. end.estimate will capture any portfolio balance updates at the
        // history start param.
        lastKnownPlanChangeAction.stateChange.end.estimate,
        planParamsHistory.slice(
          _.sortedIndexBy<{ params: { timestamp: number } }>(
            planParamsHistory,
            { params: { timestamp: lastKnownPlanChangeAction.timestamp } },
            (x) => x.params.timestamp,
          ),
        ),
      )
    },
    [planId, ianaTimezoneName, marketData, runTests],
  )
  return { applyCache }
}

const _cutDetail = (
  endTimestamp: number,
  info: PortfolioBalanceEstimation.Detail,
): PortfolioBalanceEstimation.Detail => {
  assert(endTimestamp >= info.startTimestamp)
  if (endTimestamp >= info.endTimestamp) return info

  return {
    startTimestamp: info.startTimestamp,
    endTimestamp,
    startState: info.startState,
    actions: info.actions.slice(
      0,
      _.sortedLastIndexBy<{ timestamp: number }>(
        info.actions,
        { timestamp: endTimestamp },
        (x) => x.timestamp,
      ),
    ),
  }
}

const _joinDetails = (
  startDetail: PortfolioBalanceEstimation.Detail | null,
  endDetail: PortfolioBalanceEstimation.Detail,
): PortfolioBalanceEstimation.Detail => {
  if (!startDetail) return endDetail
  assert(startDetail.actions.length > 0)
  assert(startDetail.endTimestamp === endDetail.startTimestamp)
  // May need to  allow approx floating point equality here.
  // The startState of endDetail is the same as the endState of the last action of
  // startDetail.
  assert(
    _.isEqual(
      fGet(_.last(startDetail.actions)).stateChange.end,
      endDetail.startState,
    ),
  )
  return {
    startTimestamp: startDetail.startTimestamp,
    endTimestamp: endDetail.endTimestamp,
    startState: startDetail.startState,
    actions: [...startDetail.actions, ...endDetail.actions],
  }
}

const _mergeConsecutivePlanChanges = (
  actions: PortfolioBalanceEstimation.Action[],
): PortfolioBalanceEstimation.Action[] => {
  const mergedActions: PortfolioBalanceEstimation.Action[] = []
  for (const currAction of actions) {
    const prevAction = _.last(mergedActions)
    // Don't have to avoid filtering out the first planChange because that
    // is contained in the startState of the detail.
    if (
      prevAction &&
      prevAction.args.type === 'planChange' &&
      currAction.args.type === 'planChange'
    ) {
      // Merge.
      const mergedAction: PortfolioBalanceEstimation.Action = {
        timestamp: currAction.timestamp,
        args: {
          type: 'planChange',
          allocation: currAction.args.allocation,
          portfolioUpdate:
            currAction.args.portfolioUpdate ?? prevAction.args.portfolioUpdate,
        },
        stateChange: {
          start: prevAction.stateChange.start,
          end: currAction.stateChange.end,
        },
      }
      mergedActions[mergedActions.length - 1] = mergedAction
    } else {
      mergedActions.push(currAction)
    }
  }
  return mergedActions
}

const _getServerArgs = (
  planParamsHistory: readonly { id: string; params: PlanParams }[],
  startingPortfolioBalance: number,
  estimationTimestamp: number,
  getCalendarDayFromTimestamp: (timestamp: number) => CalendarDay,
): WirePortfolioBalanceEstimationArgs => {
  // Use this only to get the withdrawalAndContributionTimes and
  // monthlyRebalanceTimes. Don't actually filter. We get all the planChange
  // actions to improve the cache hit rate.
  const { withdrawalAndContributionTimes, monthlyRebalanceTimes } =
    PlanParamsHistoryFns.filterForCurrentPortfolioBalanceEstimation({
      endTime: estimationTimestamp,
      marketCloses: 'useConservativeGuess',
      planParamsHistory: planParamsHistory.map((x) => ({
        planParamsChangeId: x.id,
        timestamp: new Date(x.params.timestamp),
      })),
    })

  const historyItemForNonMarketCloseActionsUnsorted: {
    timestampMs: number
    actionType: WirePortfolioBalanceEstimationArgsNonMarketActionType
    historyItem: { id: string; params: PlanParams }
  }[] = []
  planParamsHistory.forEach((curr, i) => {
    const nextTimestamp =
      i === planParamsHistory.length - 1
        ? Number.MAX_SAFE_INTEGER
        : planParamsHistory[i + 1].params.timestamp

    historyItemForNonMarketCloseActionsUnsorted.push({
      timestampMs: curr.params.timestamp,
      actionType:
        WirePortfolioBalanceEstimationArgsNonMarketActionType.PLAN_CHANGE,
      historyItem: curr,
    })
    while (
      withdrawalAndContributionTimes.length > 0 &&
      withdrawalAndContributionTimes[0] < nextTimestamp
    ) {
      historyItemForNonMarketCloseActionsUnsorted.push({
        timestampMs: withdrawalAndContributionTimes.shift()!,
        actionType:
          WirePortfolioBalanceEstimationArgsNonMarketActionType.WITHDRAWAL_AND_CONTRIBUTION,
        historyItem: curr,
      })
    }
    while (
      monthlyRebalanceTimes.length > 0 &&
      monthlyRebalanceTimes[0] < nextTimestamp
    ) {
      historyItemForNonMarketCloseActionsUnsorted.push({
        timestampMs: monthlyRebalanceTimes.shift()!,
        actionType:
          WirePortfolioBalanceEstimationArgsNonMarketActionType.MONTHLY_REBALANCE,
        historyItem: curr,
      })
    }
  })

  const wireArgs: WirePortfolioBalanceEstimationArgs = {
    planParamsForNonMarketActionsUnsorted:
      historyItemForNonMarketCloseActionsUnsorted.map((x) => ({
        id: x.historyItem.id,
        timestampMs: x.timestampMs,
        actionType: x.actionType,
        planParams: getPlanParamsServer(
          normalizePlanParams(x.historyItem.params, {
            timestamp: x.timestampMs,
            calendarDay: getCalendarDayFromTimestamp(x.timestampMs),
          }),
          1, // numRuns
          1, // seed does not matter because first month is deterministic.
        ),
      })),
    startBalance: startingPortfolioBalance,
  }
  return wireArgs
}

const _checkAgainstLocal = (
  remote: PortfolioBalanceEstimation.Detail,
  planId: string,
  estimationTimestamp: number,
  ianaTimezoneName: string,
  marketData: MarketData.Data,
  planParamsHistory: readonly { id: string; params: PlanParams }[],
) => {
  const local = PortfolioBalanceEstimation.getInfo(
    planId,
    planParamsHistory,
    block(() => {
      const firstParams = planParamsHistory[0].params
      assert(firstParams.wealth.portfolioBalance.isDatedPlan)
      assert(firstParams.wealth.portfolioBalance.updatedHere)
      return firstParams.wealth.portfolioBalance.amount
    }),
    estimationTimestamp,
    ianaTimezoneName,
    marketData,
    fWASM(),
  )

  const diffInfo = {
    state: {
      estimate: 0,
      allocation: 0,
    },
    planChange: {
      allocation: 0,
    },
    marketClose: {
      vt: 0,
      bnd: 0,
    },
    monthlyRebalance: {
      allocation: 0,
    },
    withdrawalAndContribution: {
      value: 0,
    },
  }
  const _state = (
    local: PortfolioBalanceEstimation.Detail['startState'],
    remote: PortfolioBalanceEstimation.Detail['startState'],
  ) => {
    const estimateDiff = Math.abs(local.estimate - remote.estimate)
    // assert(estimateDiff < 0.00001)
    const allocationDiff = Math.abs(
      local.allocation.stocks - remote.allocation.stocks,
    )
    // assert(allocationDiff < 0.00001)
    diffInfo.state.estimate = Math.max(diffInfo.state.estimate, estimateDiff)
    diffInfo.state.allocation = Math.max(
      diffInfo.state.allocation,
      allocationDiff,
    )
  }

  const _action = (
    local: PortfolioBalanceEstimation.Action,
    remote: PortfolioBalanceEstimation.Action,
  ) => {
    assert(local.timestamp === remote.timestamp)
    switch (local.args.type) {
      case 'planChange': {
        assert(remote.args.type === 'planChange')
        const allocationDiff = Math.abs(
          local.args.allocation.stocks - remote.args.allocation.stocks,
        )

        // assert(allocationDiff < 0.0001)

        diffInfo.planChange.allocation = Math.max(
          diffInfo.planChange.allocation,
          allocationDiff,
        )
        assert(
          _.isEqual(local.args.portfolioUpdate, remote.args.portfolioUpdate),
        )
        break
      }
      case 'marketClose': {
        assert(remote.args.type === 'marketClose')
        const vtDiff = Math.abs(
          local.args.marketData.percentageChangeFromLastClose.vt -
            remote.args.marketData.percentageChangeFromLastClose.vt,
        )
        // assert(vtDiff < 0.000001)
        diffInfo.marketClose.vt = Math.max(diffInfo.marketClose.vt, vtDiff)
        const bndDiff = Math.abs(
          local.args.marketData.percentageChangeFromLastClose.bnd -
            remote.args.marketData.percentageChangeFromLastClose.bnd,
        )
        // assert(bndDiff < 0.000001)
        diffInfo.marketClose.bnd = Math.max(diffInfo.marketClose.bnd, bndDiff)
        break
      }
      case 'monthlyRebalance':
        assert(remote.args.type === 'monthlyRebalance')
        const allocationDiff = Math.abs(
          local.args.allocation.stocks - remote.args.allocation.stocks,
        )
        // assert(allocationDiff < 0.000001)
        diffInfo.monthlyRebalance.allocation = Math.max(
          diffInfo.monthlyRebalance.allocation,
          allocationDiff,
        )
        break
      case 'withdrawalAndContribution': {
        assert(remote.args.type === 'withdrawalAndContribution')
        assert(
          local.args.netContributionOrWithdrawal.type ===
            remote.args.netContributionOrWithdrawal.type,
        )
        const localValue =
          local.args.netContributionOrWithdrawal.type === 'withdrawal'
            ? -local.args.netContributionOrWithdrawal.withdrawal
            : local.args.netContributionOrWithdrawal.contribution
        const remoteValue =
          remote.args.netContributionOrWithdrawal.type === 'withdrawal'
            ? -remote.args.netContributionOrWithdrawal.withdrawal
            : remote.args.netContributionOrWithdrawal.contribution
        const valueDiff = Math.abs(localValue - remoteValue)
        // assert(valueDiff < 0.00001)
        diffInfo.withdrawalAndContribution.value = Math.max(
          diffInfo.withdrawalAndContribution.value,
          valueDiff,
        )
        break
      }
      default:
        noCase(local.args)
    }
    _state(local.stateChange.start, remote.stateChange.start)
    _state(local.stateChange.end, remote.stateChange.end)
  }

  assert(local.startTimestamp === remote.startTimestamp)
  assert(local.endTimestamp === remote.endTimestamp)
  _state(local.startState, remote.startState)
  assert(local.actions.length === remote.actions.length)
  local.actions.forEach((x, i) => _action(x, remote.actions[i]))
  const diffArray = [
    diffInfo.state.estimate,
    diffInfo.state.allocation,
    diffInfo.planChange.allocation,
    diffInfo.marketClose.vt,
    diffInfo.marketClose.bnd,
    diffInfo.monthlyRebalance.allocation,
    diffInfo.withdrawalAndContribution.value,
  ]

  const maxDiff = Math.max(...diffArray)

  console.log(
    `%c${maxDiff}`,
    maxDiff < 0.00001
      ? 'font-weight:bold;color:#16a34a;border-radius:5px;padding:5px'
      : 'font-weight:bold;color:#e11d48;border-radius:5px;padding:5px',
    'maxDiff:PortfolioBalanceEstimation',
    // {
    //   diffInfo,
    //   local,
    //   remote,
    // },
  )
}
