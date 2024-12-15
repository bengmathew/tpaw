import * as Sentry from '@sentry/nextjs'
import {
  assert,
  assertFalse,
  block,
  fGet,
  getZonedTimeFns,
  MarketData,
  noCase,
  PlanParams,
  PlanParamsHistoryFns
} from '@tpaw/common'
import _ from 'lodash'
import * as uuid from 'uuid'
import {
  normalizePlanParams,
  PlanParamsNormalized,
} from '../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { CallRust } from '../../../Simulator/PlanParamsProcessed/CallRust'
import { processPlanParams } from '../../../Simulator/PlanParamsProcessed/PlanParamsProcessed'
import {
  FirstMonthSavingsPortfolioDetail,
  getFirstMonthSavingsPortfolioDetail,
} from '../../../Simulator/Simulator/GetFirstMonthSavingsPortfolioDetail'
import { WASM } from '../../../Simulator/Simulator/GetWASM'
import { runSimulationInWASM } from '../../../Simulator/Simulator/RunSimulationInWASM'
import { CalendarDayFns } from '../../../Utils/CalendarDayFns'
import { groupBy } from '../../../Utils/GroupBy'
import { getMarketDataIndexForTime } from '../../Common/GetMarketData'

// TODO: Testing. Delete
export namespace PortfolioBalanceEstimation {
  type _State = { estimate: number; allocation: { stocks: number } }
  export type Action = {
    timestamp: number
    args:
      | {
          type: 'marketClose'
          marketData: {
            percentageChangeFromLastClose: MarketData.DailyStockMarketPerformance['percentageChangeFromLastClose']
          }
        }
      | {
          type: 'withdrawalAndContribution'
          netContributionOrWithdrawal: FirstMonthSavingsPortfolioDetail['contributionToOrWithdrawalFromSavingsPortfolio']
        }
      | {
          type: 'monthlyRebalance'
          allocation: { stocks: number }
        }
      | {
          type: 'planChange'
          allocation: { stocks: number }
          portfolioUpdate: { amount: number; exactTimestamp: number } | null
        }
    stateChange: {
      start: _State
      end: _State
    }
  }

  type _ActionSpec = {
    timestamp: number
    getArgs: (currentEstimate: number) => Action['args']
  }

  export type Detail = Omit<ReturnType<typeof getInfo>, 'ianaTimezoneName'>
  type DetailOld = ReturnType<typeof getInfo>

  export function getInfo(
    planId: string,
    planParamsHistory: readonly { id: string; params: PlanParams }[],
    portfolioBalanceAtHistoryStart: number,
    estimationTimestamp: number,
    ianaTimezoneName: string,
    marketData: MarketData.Data,
    wasm: WASM,
  ) {
    const startingParams = fGet(_.first(planParamsHistory)).params
    assert(startingParams.timestamp <= estimationTimestamp)

    const simulate_inOrder = _getSimulate_InOrder(
      planId,
      planParamsHistory,
      marketData,
      wasm,
      ianaTimezoneName,
    )
    const getAllocationForParams = (params: PlanParams, estimate: number) =>
      // params.results?.displayedAssetAllocation ??
      simulate_inOrder(params.timestamp, estimate).afterWithdrawals.allocation

    // --------------------
    // Market Close Actions
    // --------------------

    const marketCloseActionSpecs: _ActionSpec[] = marketData
      // If end <= start, then the array is empty.
      .slice(
        getMarketDataIndexForTime(startingParams.timestamp, marketData) + 1,
        getMarketDataIndexForTime(estimationTimestamp, marketData) + 1,
      )
      .map((x) => x.dailyStockMarketPerformance)
      .map((x) => ({
        timestamp: x.closingTime,
        getArgs: () => ({ type: 'marketClose', marketData: x }),
      }))

    const {
      withdrawalAndContributionTimes,
      monthlyRebalanceTimes,
      startTime,
      idsToDelete,
    } = PlanParamsHistoryFns.filterForCurrentPortfolioBalanceEstimation({
      endTime: estimationTimestamp,
      marketCloses: marketCloseActionSpecs.map((x) => x.timestamp),
      planParamsHistory: planParamsHistory.map((x) => ({
        planParamsChangeId: x.id,
        timestamp: new Date(x.params.timestamp),
      })),
    })
    // We don't delete on server.
    const planParamsHistoryFiltered = planParamsHistory.slice()
    // const planParamsHistoryFiltered = planParamsHistory.filter(
    //   (x) => !idsToDelete.has(x.id),
    // )

    // -------------------
    // Withdarwal And Contribution Actions
    // -------------------
    const withdrawalAndContributionActionSpecs: _ActionSpec[] =
      withdrawalAndContributionTimes.map((timestamp) => {
        return {
          timestamp,
          getArgs: (estimate) => ({
            type: 'withdrawalAndContribution',
            netContributionOrWithdrawal: simulate_inOrder(timestamp, estimate)
              .contributionToOrWithdrawalFromSavingsPortfolio,
          }),
        }
      })
    // -------------------
    // Monthly Rebalance Actions
    // -------------------
    const monthlyRebalanceActionSpecs: _ActionSpec[] =
      monthlyRebalanceTimes.map((timestamp) => {
        return {
          timestamp,
          getArgs: (estimate) => ({
            type: 'monthlyRebalance',
            allocation: simulate_inOrder(timestamp, estimate).afterWithdrawals
              .allocation,
          }),
        }
      })

    // -------------------
    // Plan Change Actions
    // -------------------
    const planChangeActionSpecs = _.compact(
      planParamsHistoryFiltered.map(({ params }, i): _ActionSpec | null => {
        if (i === 0) {
          assert(params === startingParams)
          return null
        }
        const prev = planParamsHistoryFiltered[i - 1]
        const prevPortfolioBalanceChangedAtId =
          !prev.params.wealth.portfolioBalance.isDatedPlan ||
          prev.params.wealth.portfolioBalance.updatedHere
            ? prev.id
            : prev.params.wealth.portfolioBalance.updatedAtId
        const timestamp = params.timestamp
        const portfolioUpdate =
          !params.wealth.portfolioBalance.isDatedPlan ||
          params.wealth.portfolioBalance.updatedHere
            ? {
                amount: params.wealth.portfolioBalance.amount,
                exactTimestamp: params.timestamp,
              }
            : params.wealth.portfolioBalance.updatedAtId !==
                prevPortfolioBalanceChangedAtId
              ? {
                  amount: params.wealth.portfolioBalance.updatedTo,
                  exactTimestamp:
                    params.wealth.portfolioBalance.updatedAtTimestamp,
                }
              : null
        return {
          timestamp,
          getArgs: (currEstimate) => {
            return {
              type: 'planChange',
              allocation: getAllocationForParams(
                params,
                portfolioUpdate?.amount ?? currEstimate,
              ),
              portfolioUpdate,
            }
          },
        }
      }),
    )

    // -------------------
    const actionSpecs: _ActionSpec[] =
      PlanParamsHistoryFns.combineActionsForPortfolioBalanceEstimation<_ActionSpec>(
        {
          marketCloseActions: marketCloseActionSpecs,
          monthlyRebalanceActions: monthlyRebalanceActionSpecs,
          planChangeActions: planChangeActionSpecs,
          withdrawalAndContributionActions:
            withdrawalAndContributionActionSpecs,
        },
        (x) => x.timestamp,
      )

    const start = {
      amount: portfolioBalanceAtHistoryStart,
      timestamp: startTime,
      allocation: getAllocationForParams(
        startingParams,
        portfolioBalanceAtHistoryStart,
      ),
    }
    let currState = { estimate: start.amount, allocation: start.allocation }
    let actions = [] as Action[]

    while (actionSpecs.length > 0) {
      actions.push(_applyAction(fGet(actionSpecs.shift()), currState))
      currState = fGet(_.last(actions)).stateChange.end
    }

    // Actions can be empty, but if they are not, they are all between
    // start.timestamp and estimationTimestamp (inclusive).
    if (actions.length > 0) {
      const firstAction = fGet(_.first(actions))
      if (firstAction.timestamp < start.timestamp) {
        Sentry.captureMessage(
          `First action too early: ${JSON.stringify(firstAction)}\n start timestamp: ${start.timestamp}\n startingParams.timestamp: ${startingParams.timestamp}\n estimationTimestamp: ${estimationTimestamp}`,
        )
        assertFalse()
      }
      assert(fGet(_.last(actions)).timestamp <= estimationTimestamp)
    }

    return {
      ianaTimezoneName,
      startTimestamp: start.timestamp,
      endTimestamp: estimationTimestamp,
      startState: { estimate: start.amount, allocation: start.allocation },
      actions,
    }
  }

  const _applyAction = (
    { timestamp, getArgs }: _ActionSpec,
    start: _State,
  ): Action => {
    let st = performance.now()
    const args = getArgs(start.estimate)
    st = performance.now()
    const endFull = block<_State>(() => {
      switch (args.type) {
        case 'marketClose':
          const result = {
            estimate:
              (1 + args.marketData.percentageChangeFromLastClose.vt) *
                start.estimate *
                start.allocation.stocks +
              (1 + args.marketData.percentageChangeFromLastClose.bnd) *
                start.estimate *
                (1 - start.allocation.stocks),
            allocation: start.allocation,
          }
          return result
        case 'withdrawalAndContribution':
          return {
            estimate:
              start.estimate +
              (args.netContributionOrWithdrawal.type === 'contribution'
                ? args.netContributionOrWithdrawal.contribution
                : -args.netContributionOrWithdrawal.withdrawal),
            allocation: start.allocation,
          }
        case 'monthlyRebalance':
          return {
            estimate: start.estimate,
            allocation: args.allocation,
          }
        case 'planChange':
          return {
            estimate: args.portfolioUpdate?.amount ?? start.estimate,
            allocation: args.allocation,
          }
        default:
          noCase(args)
      }
    })
    const end = {
      estimate: endFull.estimate,
      allocation: endFull.allocation,
    }

    return { timestamp, args, stateChange: { start, end } }
  }

  let timing = {
    t1: 0,
    t2: 0,
    t3: 0,
  }

  const simulationCache: {
    planId: string
    map: Map<string, FirstMonthSavingsPortfolioDetail>
  } = {
    planId: uuid.v4(),
    map: new Map(),
  }
  const _getSimulate_InOrder = (
    planId: string,
    planParamsList: readonly {
      id: string
      params: PlanParams
    }[],
    marketData: MarketData.Data,
    wasm: WASM,
    ianaTimezoneName: string,
  ) => {
    const pickPlanParamsForTimestamp = _getPickDataForTime_InOrder(
      planParamsList,
      (x) => x.params.timestamp,
    )
    const pickMarketDataForTimestamp = _getPickDataForTime_InOrder(
      marketData,
      (x) => x.closingTime,
    )
    return (timestamp: number, portfolioBalance: number) => {
      if (planId !== simulationCache.planId) {
        simulationCache.planId = planId
        simulationCache.map.clear()
      }

      const historyItem = pickPlanParamsForTimestamp(timestamp)
      const marketDataAtTime = {
        ...pickMarketDataForTimestamp(timestamp),
        timestampForMarketData: timestamp,
      }

      const cacheKey = `${historyItem.id}/${timestamp}/${portfolioBalance}`

      const planParams = historyItem.params
      const planParamsNorm = normalizePlanParams(planParams, {
        timestamp,
        calendarDay: CalendarDayFns.fromTimestamp(timestamp, ianaTimezoneName),
      })

      let start = performance.now()
      start = performance.now()
      timing.t1 += performance.now() - start
      const planParamsRust = CallRust.getPlanParamsRust(planParamsNorm)
      const planParamsProcessed = processPlanParams(
        planParamsNorm,
        marketDataAtTime,
      )
      timing.t2 += performance.now() - start
      start = performance.now()

      const result = getFirstMonthSavingsPortfolioDetail(
        runSimulationInWASM(
          portfolioBalance,
          planParamsRust,
          marketDataAtTime,
          planParamsNorm,
          planParamsProcessed,
          { start: 0, end: 1 },
          0, // Does not matter since first month is deterministic.
          wasm,
          {
            forFirstMonth: true,
          },
        ).byMonthsFromNowByRun.savingsPortfolio,
        planParamsProcessed,
      )
      timing.t3 += performance.now() - start
      start = performance.now()

      simulationCache.map.set(cacheKey, result)
      return result
    }
  }

  // This is much faster than _.sortedIndex in _getDataIndexForTime() when data gets long.
  const _getPickDataForTime_InOrder = <T>(
    data: readonly T[],
    getTime: (t: T) => number,
  ) => {
    const dataCopy = data.map((x) => ({ timestamp: getTime(x), value: x }))
    let prevTimestamp = null as null | number
    return (timestamp: number): T => {
      assert(timestamp >= (prevTimestamp ?? timestamp))
      prevTimestamp = timestamp
      assert(dataCopy[0].timestamp <= timestamp)
      while (dataCopy.length > 1 && dataCopy[1].timestamp <= timestamp) {
        dataCopy.shift()
      }
      return dataCopy[0].value
    }
  }

  export type ByMonthInfo = ReturnType<typeof getByMonthDetail>

  // Note, this can be empty since info.actions can be empty.
  export const getByMonthDetail = (
    detail: Detail,
    ianaTimezoneName: string,
  ) => {
    const result = (x: typeof monthsDesc) => ({ ...detail, monthsDesc: x })

    if (detail.actions.length === 0) return result([])
    const getZonedTime = getZonedTimeFns(ianaTimezoneName)
    const actionsDesc = _.reverse(detail.actions.slice())

    const actionsDescByMonthMap = groupBy(actionsDesc, (x) =>
      getZonedTime(x.timestamp).startOf('month').toMillis(),
    )

    let currState = fGet(_.first(actionsDesc)).stateChange.end

    const monthsDesc = _getDescTimeSeq(
      ianaTimezoneName,
      detail.endTimestamp,
      detail.startTimestamp,
      'month',
    )
      .map((month) => {
        const actionsDesc = actionsDescByMonthMap.get(month) ?? []
        actionsDescByMonthMap.delete(month)
        return { month, actionsDesc }
      })
      .map(({ month, actionsDesc }, i) => {
        const actionsDescByDayMap = groupBy(actionsDesc, (action) =>
          getZonedTime(action.timestamp).startOf('day').toMillis(),
        )

        const daysDesc = _getDescTimeSeq(
          ianaTimezoneName,
          Math.min(
            getZonedTime(month).endOf('month').toMillis(),
            detail.endTimestamp,
          ),
          Math.max(month, detail.startTimestamp),
          'day',
        ).map((day) => {
          const actionsDesc = actionsDescByDayMap.get(day) ?? []
          actionsDescByDayMap.delete(day)

          return { day, actionsDesc }
        })
        assert(actionsDescByDayMap.size === 0)

        const stateChange = {
          end: currState,
          start: _.last(actionsDesc)?.stateChange.start ?? currState,
        }
        currState = stateChange.start
        return { month, stateChange, daysDesc }
      })
    assert(actionsDescByMonthMap.size === 0)
    return result(monthsDesc)
  }

  export type EstimateDetailFixed = ReturnType<
    typeof addCurrentParamsAfterLastActionInEstimateDetail
  >
  export const addCurrentParamsAfterLastActionInEstimateDetail = (
    estimateDetail: PortfolioBalanceEstimation.Detail,
    currentParams: PlanParamsNormalized,
  ) => {
    const { datingInfo } = currentParams
    const { portfolioBalance } = currentParams.wealth
    assert(datingInfo.isDated)
    assert(portfolioBalance.isDatedPlan)
    const lastActionTimestamp =
      _.last(estimateDetail.actions)?.timestamp ?? estimateDetail.startTimestamp
    assert(lastActionTimestamp < datingInfo.nowAsTimestamp)
    assert(estimateDetail.endTimestamp >= datingInfo.nowAsTimestamp)

    const portfolioUpdate = portfolioBalance.updatedHere
      ? {
          amount: portfolioBalance.amount,
          exactTimestamp: datingInfo.nowAsTimestamp,
        }
      : portfolioBalance.updatedAtTimestamp > lastActionTimestamp
        ? {
            amount: portfolioBalance.updatedTo,
            exactTimestamp: portfolioBalance.updatedAtTimestamp,
          }
        : null

    const prevState =
      _.last(estimateDetail.actions)?.stateChange.end ??
      estimateDetail.startState

    return {
      ...estimateDetail,
      actions: [
        ...estimateDetail.actions,
        {
          timestamp: datingInfo.nowAsTimestamp,
          args: {
            type: 'planChange' as const,
            stockAllocation: 'current' as const,
            portfolioUpdate,
          },
          stateChange: {
            start: prevState,
            end: {
              estimate: portfolioUpdate?.amount ?? prevState.estimate,
              allocation: 'current' as const,
            },
          },
        },
      ],
    }
  }

  const _getDescTimeSeq = (
    ianaTimezoneName: string,
    end: number,
    startIn: number,
    delta: 'day' | 'month',
  ) => {
    const getZonedTime = getZonedTimeFns(ianaTimezoneName)
    const result = [] as number[]
    const start = getZonedTime(startIn).startOf(delta)
    let curr = getZonedTime(end).startOf(delta)
    while (curr >= start) {
      result.push(curr.toMillis())
      curr = curr.minus({ [delta]: 1 })
    }
    return result
  }
}
