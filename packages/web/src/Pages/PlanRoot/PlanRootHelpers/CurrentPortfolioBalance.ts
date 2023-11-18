import * as Sentry from '@sentry/nextjs'
import {
  MarketData,
  PlanParams,
  PlanParamsHistoryFns,
  assert,
  block,
  fGet,
  getZonedTimeFns,
  letIn,
  noCase,
} from '@tpaw/common'
import { getAuth } from 'firebase/auth'
import _ from 'lodash'
import * as uuid from 'uuid'
import { extendPlanParams } from '../../../TPAWSimulator/ExtentPlanParams'
import { processPlanParams } from '../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import {
  FirstMonthSavingsPortfolioDetail,
  firstMonthSavingsPortfolioDetail,
} from '../../../TPAWSimulator/Worker/FirstMonthSavingsPortfolioDetail'
import { WASM } from '../../../TPAWSimulator/Worker/GetWASM'
import { runSimulationInWASM } from '../../../TPAWSimulator/Worker/RunSimulationInWASM'
import { groupBy } from '../../../Utils/GroupBy'
import { getMarketDataIndexForTime } from '../../Common/GetMarketData'

export namespace CurrentPortfolioBalance {
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
          type: 'withdarwalAndContribution'
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

  export type Info = ReturnType<typeof getInfo>

  export function getInfo(
    planId: string,
    planParamsHistory: readonly { id: string; params: PlanParams }[],
    estimationTimestamp: number,
    ianaTimezoneName: string,
    marketData: MarketData.Data,
    wasm: WASM,
  ) {
    const startingParams = fGet(_.first(planParamsHistory)).params
    assert(startingParams.timestamp <= estimationTimestamp)
    const startingPortfolioBalance = startingParams.wealth.portfolioBalance
    assert(startingPortfolioBalance.updatedHere)

    const simulate_inOrder = _getSimulate_InOrder(
      planId,
      planParamsHistory,
      marketData,
      wasm,
      ianaTimezoneName,
    )
    const getAllocationForParams = (params: PlanParams, estimate: number) =>
      params.results?.displayedAssetAllocation ??
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
    const planParamsHistoryFiltered = planParamsHistory.filter(
      (x) => !idsToDelete.has(x.id),
    )

    const start = block(() => {
      const amount = startingPortfolioBalance.amount
      const timestamp = startTime
      const allocation = getAllocationForParams(startingParams, amount)
      return { amount, timestamp, allocation }
    })

    // -------------------
    // Withdarwal And Contribution Actions
    // -------------------
    const withdrawalAndContributionActionSpecs: _ActionSpec[] =
      withdrawalAndContributionTimes.map((timestamp) => {
        return {
          timestamp,
          getArgs: (estimate) => ({
            type: 'withdarwalAndContribution',
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
        const prevPortfolioBalanceChangedAtId = prev.params.wealth
          .portfolioBalance.updatedHere
          ? prev.id
          : prev.params.wealth.portfolioBalance.updatedAtId
        const timestamp = params.timestamp
        const portfolioUpdate = params.wealth.portfolioBalance.updatedHere
          ? {
              amount: params.wealth.portfolioBalance.amount,
              exactTimestamp: params.timestamp,
            }
          : params.wealth.portfolioBalance.updatedAtId !==
            prevPortfolioBalanceChangedAtId
          ? {
              amount: params.wealth.portfolioBalance.updatedTo,
              exactTimestamp: params.wealth.portfolioBalance.updatedAtTimestamp,
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

    let currState = { estimate: start.amount, allocation: start.allocation }
    let actions = [] as Action[]

    while (actionSpecs.length > 0) {
      actions.push(_applyAction(fGet(actionSpecs.shift()), currState))
      currState = fGet(_.last(actions)).stateChange.end
    }

    // Actions can be empty, but if they are not, they are all between
    // start.timestamp and estimationTimestamp (inclusive).
    if (actions.length > 0) {
      assert(fGet(_.first(actions)).timestamp >= start.timestamp)
      assert(fGet(_.last(actions)).timestamp <= estimationTimestamp)
    }
    const result = {
      ianaTimezoneName,
      startTimestamp: start.timestamp,
      endTimestamp: estimationTimestamp,
      startState: { estimate: start.amount, allocation: start.allocation },
      actions,
    }
    return result
  }

  const _applyAction = (
    { timestamp, getArgs }: _ActionSpec,
    start: _State,
  ): Action => {
    let st = performance.now()
    const args = getArgs(start.estimate)
    st = performance.now()
    const end = block<_State>(() => {
      switch (args.type) {
        case 'marketClose':
          return {
            estimate:
              (1 + args.marketData.percentageChangeFromLastClose.vt) *
                start.estimate *
                start.allocation.stocks +
              (1 + args.marketData.percentageChangeFromLastClose.bnd) *
                start.estimate *
                (1 - start.allocation.stocks),
            allocation: start.allocation,
          }
        case 'withdarwalAndContribution':
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
      const marketDataAtTime = pickMarketDataForTimestamp(timestamp)

      const cacheKey = `${historyItem.id}/${timestamp}/${portfolioBalance}`
      const fromCache = simulationCache.map.get(cacheKey)
      if (fromCache) return fromCache

      const planParams = historyItem.params

      let start = performance.now()
      const paramsExt = extendPlanParams(
        planParams,
        timestamp,
        ianaTimezoneName,
      )
      start = performance.now()
      timing.t1 += performance.now() - start
      const paramsProcessed = processPlanParams(
        paramsExt,
        portfolioBalance,
        marketDataAtTime,
      )
      timing.t2 += performance.now() - start
      start = performance.now()

      const result = firstMonthSavingsPortfolioDetail(
        runSimulationInWASM(paramsProcessed, { start: 0, end: 1 }, wasm, {
          forFirstMonth: true,
        }).byMonthsFromNowByRun.savingsPortfolio,
        paramsProcessed,
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

  export type ByMonthInfo = ReturnType<typeof getByMonthInfo>

  // Note, this can be empty since info.actions can be empty.
  export const getByMonthInfo = (info: Info) => {
    const result = (x: typeof monthsDesc) => ({ ...info, monthsDesc: x })

    if (info.actions.length === 0) return result([])
    const getZonedTime = getZonedTimeFns(info.ianaTimezoneName)
    const actionsDesc = _.reverse(info.actions.slice())

    const actionsDescByMonthMap = groupBy(actionsDesc, (x) =>
      getZonedTime(x.timestamp).startOf('month').toMillis(),
    )

    let currState = fGet(_.first(actionsDesc)).stateChange.end

    const monthsDesc = _getDescTimeSeq(
      info.ianaTimezoneName,
      info.endTimestamp,
      info.startTimestamp,
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
          info.ianaTimezoneName,
          Math.min(
            getZonedTime(month).endOf('month').toMillis(),
            info.endTimestamp,
          ),
          Math.max(month, info.startTimestamp),
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

  export const mergeByMonthInfo = (...parts: ByMonthInfo[]) =>
    parts.reduce((left: ByMonthInfo, right: ByMonthInfo): ByMonthInfo => {
      assert(left.ianaTimezoneName === right.ianaTimezoneName)
      assert(left.endTimestamp === right.startTimestamp)
      if (left.actions.length === 0) {
        assert(_.isEqual(left.startState, right.startState))
        return right
      }

      // Don't need to check right.action.length === 0, the following code
      // handles that case as well.

      const lastLeftMonth = fGet(_.first(left.monthsDesc))
      const firstRightMonth = fGet(_.last(right.monthsDesc))
      assert(lastLeftMonth.month === firstRightMonth.month)
      const lastLeftDay = fGet(_.first(lastLeftMonth.daysDesc))
      const firstRightDay = fGet(_.last(firstRightMonth.daysDesc))
      assert(lastLeftDay.day === firstRightDay.day)
      const [
        lastLeftDayActionsDescAtEndTimestamp,
        lastLeftDayActionsDescBeforeEndTimestamp,
      ] = _.partition(
        lastLeftDay.actionsDesc,
        (x) => x.timestamp === left.endTimestamp,
      )

      // fGet because when merging left has to have endTimestamp be at the last
      // action, which should be a plan change action. _.last because plan change
      // actions are always first (so last in desc order) for actions with same timestamps.
      const commonPlanChangeAction = fGet(
        _.last(lastLeftDayActionsDescAtEndTimestamp),
      )
      assert(commonPlanChangeAction.args.type === 'planChange')
      assert(
        _.isEqual(commonPlanChangeAction.stateChange.end, right.startState),
      )
      assert(
        _.initial(lastLeftDayActionsDescAtEndTimestamp).every((x, i) =>
          _.isEqual(x, fGet(_.nth(firstRightDay.actionsDesc, i))),
        ),
      )

      const mergedDay: ByMonthInfo['monthsDesc'][0]['daysDesc'][0] = {
        day: lastLeftDay.day,
        actionsDesc: [
          ...firstRightDay.actionsDesc,
          commonPlanChangeAction,
          ...lastLeftDayActionsDescBeforeEndTimestamp,
        ],
      }

      assert(
        _.isEqual(
          mergedDay.actionsDesc,
          _.sortBy(mergedDay.actionsDesc, (x) => -x.timestamp),
        ),
      )

      const mergedMonth: ByMonthInfo['monthsDesc'][number] = {
        month: lastLeftMonth.month,
        stateChange: {
          start: lastLeftMonth.stateChange.start,
          end: firstRightMonth.stateChange.end,
        },
        daysDesc: [
          ..._.initial(firstRightMonth.daysDesc),
          mergedDay,
          ..._.tail(lastLeftMonth.daysDesc),
        ],
      }

      return {
        ianaTimezoneName: left.ianaTimezoneName,
        startTimestamp: left.startTimestamp,
        endTimestamp: right.endTimestamp,
        startState: left.startState,
        actions: [
          ...left.actions.slice(
            0,
            _.findLastIndex(
              left.actions,
              (x) => x.timestamp < left.endTimestamp,
            ) + 1, // -1 + 1 = 0, so [] if -1.
          ),
          ...right.actions,
        ],
        monthsDesc: [
          ..._.initial(right.monthsDesc),
          mergedMonth,
          ..._.tail(left.monthsDesc),
        ],
      }
    })

  export const getEstimateInfo = (info: ReturnType<typeof cutInfo>) => {
    const targetInfo = info.postBase ? info.postBase : info.preBase
    if (targetInfo.actions.length === 0) return null

    const lastPortfolioUpdateIndex = _.findLastIndex(
      targetInfo.actions,
      (action) =>
        action.args.type === 'planChange' &&
        action.args.portfolioUpdate !== null,
    )
    if (lastPortfolioUpdateIndex === targetInfo.actions.length - 1) return null
    if (lastPortfolioUpdateIndex === -1)
      return {
        lastEnteredAmount: targetInfo.startState.estimate,
        lastEnteredTimestamp: targetInfo.startTimestamp,
      }
    const { args } = targetInfo.actions[lastPortfolioUpdateIndex]
    assert(args.type === 'planChange' && args.portfolioUpdate)
    return {
      lastEnteredAmount: args.portfolioUpdate.amount,
      lastEnteredTimestamp: args.portfolioUpdate.exactTimestamp,
    }
  }

  export const cutInfo = (
    endTimestamp: number,
    info: { preBase: ByMonthInfo | null; postBase: Info },
  ) => {
    const cutPostBase = _cutInfo(endTimestamp, info.postBase)
    try {
      return cutPostBase
        ? ({ preBase: info.preBase, postBase: cutPostBase } as const)
        : ({
            preBase: fGet(_cutByMonthInfo(endTimestamp, fGet(info.preBase))),
            postBase: null,
          } as const)
    } catch (e) {
      getAuth()
      Sentry.captureException(e)
      Sentry.captureMessage(`
      uid: ${getAuth().currentUser?.uid ?? 'none'}
      endTimestamp: ${endTimestamp}
      info.preBase === null: ${info.preBase === null}
      info.postBase: ${
        info.postBase
          ? `
          startTimestamp: ${info.postBase.startTimestamp}
          endTimestamp: ${info.postBase.endTimestamp}
          `
          : 'null'
      }
      cutPostBase === null: ${cutPostBase === null}
      `)
      throw e
    }
  }

  const _cutInfo = (endTimestamp: number, info: Info): Info | null => {
    assert(endTimestamp <= info.endTimestamp)
    if (endTimestamp < info.startTimestamp) return null
    if (endTimestamp === info.endTimestamp) return info
    return {
      ianaTimezoneName: info.ianaTimezoneName,
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

  const _cutByMonthInfo = (
    endTimestamp: number,
    info: ByMonthInfo,
  ): ByMonthInfo | null => {
    assert(endTimestamp <= info.endTimestamp)
    if (endTimestamp < info.startTimestamp) return null

    const getZonedTime = getZonedTimeFns(info.ianaTimezoneName)
    const endDateTime = getZonedTime(endTimestamp)

    const { endMonth, prevMonthsDesc } = block(() => {
      const endMonthTimestamp = endDateTime.startOf('month').valueOf()
      const endMonthIndex = _.sortedIndexBy<{ month: number }>(
        info.monthsDesc,
        { month: endMonthTimestamp },
        (x) => -x.month,
      )
      assert(endMonthIndex !== -1)
      const endMonth = info.monthsDesc[endMonthIndex]
      assert(endMonth.month === endMonthTimestamp)
      const prevMonthsDesc = info.monthsDesc.slice(endMonthIndex + 1)
      return { endMonth, prevMonthsDesc }
    })

    const { prevDaysDesc, endDay } = block(() => {
      const endDayTimestamp = endDateTime.startOf('day').valueOf()
      const endDayIndex = _.sortedIndexBy<{ day: number }>(
        endMonth.daysDesc,
        { day: endDayTimestamp },
        (x) => -x.day,
      )
      assert(endDayIndex !== -1)
      const endDay = endMonth.daysDesc[endDayIndex]
      assert(endDay.day === endDayTimestamp)
      const prevDaysDesc = endMonth.daysDesc.slice(endDayIndex + 1)
      return { prevDaysDesc, endDay }
    })

    const { actionsDesc } = block(() => {
      const endActionIndex = _.sortedIndexBy<{ timestamp: number }>(
        endDay.actionsDesc,
        { timestamp: endDateTime.valueOf() },
        (x) => -x.timestamp,
      )
      const actionsDesc = endDay.actionsDesc.slice(endActionIndex)
      return { actionsDesc }
    })

    const actions = info.actions.slice(
      0,
      _.sortedLastIndexBy<{ timestamp: number }>(
        info.actions,
        { timestamp: endTimestamp },
        (x) => x.timestamp,
      ),
    )

    const cutDay = { day: endDay.day, actionsDesc }
    const cutMonth = {
      month: endMonth.month,
      stateChange: {
        start: endMonth.stateChange.start,
        end: _.last(actions)?.stateChange.end ?? endMonth.stateChange.start,
      },
      daysDesc: [cutDay, ...prevDaysDesc],
    }

    return {
      ianaTimezoneName: info.ianaTimezoneName,
      startTimestamp: info.startTimestamp,
      endTimestamp,
      startState: info.startState,
      actions,
      monthsDesc: [cutMonth, ...prevMonthsDesc],
    }
  }

  export const get = (info: ReturnType<typeof cutInfo>): number =>
    letIn(
      info.postBase ?? info.preBase,
      (data) => _.last(data.actions)?.stateChange.end ?? data.startState,
    ).estimate

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
