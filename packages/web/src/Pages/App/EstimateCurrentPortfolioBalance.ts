import { Params, PlanParams, assert, fGet, noCase } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { extendParams } from '../../TPAWSimulator/ExtentParams'
import { processPlanParams } from '../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import {
  FirstMonthSavingsPortfolioDetail,
  firstMonthSavingsPortfolioDetail,
} from '../../TPAWSimulator/Worker/FirstMonthSavingsPortfolioDetail'
import { WASM } from '../../TPAWSimulator/Worker/GetWASM'
import { runSimulationInWASM } from '../../TPAWSimulator/Worker/RunSimulationInWASM'
import { getDateTimeInSameTimeZone } from '../../Utils/GetDateTimeInSameTimeZone'
import { MarketData } from '../Common/GetMarketData'

type _State = { estimate: number; allocation: { stocks: number } }
export type CurrentPortfolioBalanceEstimateAction = {
  time: number
  args:
    | {
        type: 'marketClose'
        marketData: {
          percentageChangeFromLastClose: MarketData['dailyStockMarketPerformance'][0]['percentageChangeFromLastClose']
        }
      }
    | {
        type: 'monthBoundary'
        netContributionOrWithdrawal: FirstMonthSavingsPortfolioDetail['contributionToOrWithdrawalFromSavingsPortfolio']
        allocation: { stocks: number }
      }
    | {
        type: 'planChangeRebalance'
        allocation: { stocks: number }
      }
  stateChange: {
    start: _State
    end: _State
  }
}

type _ActionSpec = {
  time: number
  getArgs: (
    currentEstimate: number,
  ) => CurrentPortfolioBalanceEstimateAction['args']
}

export type CurrentPortfolioBalanceEstimateInfo = ReturnType<
  typeof estimateCurrentPortfolioBalance
>
export function estimateCurrentPortfolioBalance(
  params: Params,
  currentTime: DateTime,
  marketData: MarketData,
  wasm: WASM,
) {
  const { portfolioBalance } = params.plan.wealth

  const getDateTimeInCurrentTimezone = getDateTimeInSameTimeZone(currentTime)
  const simulate = _getSimulate(
    params,
    marketData,
    getDateTimeInCurrentTimezone,
    wasm,
  )
  const getAllocationAndWithdrawalForMonthBoundary = (
    time: number,
    estimate: number,
  ): {
    netContributionOrWithdrawal:
      | { type: 'contribution'; contribution: number }
      | { type: 'withdrawal'; withdrawal: number }
    allocation: { stocks: number }
  } => {
    if (time < params.plan.timestamp) {
      assert(!portfolioBalance.isLastPlanChange)
      // +1 to remove doubt about wether this will return the start of month of
      // end of prev month.
      const { year, month } = getDateTimeInCurrentTimezone.fromMillis(time + 1)
      // FEATURE: There is a small window where params was changed right before
      // the end of the month and then user came back in a different timzone and
      // and pushed param change to the start of the next month. Now we will
      // need to get month boundary from the saved list but it won't be there,
      // because when it was saved in the old timezone, the month boundary was
      // not hit. Solve this simulating at param timestamp. It should be close
      // enough.
      const { allocation, netContributionOrWithdrawal } = fGet(
        _.find(
          portfolioBalance.history.monthBoundaryDetails,
          (x) => x.startOfMonth.year === year && x.startOfMonth.month === month,
        ),
      )
      return { allocation, netContributionOrWithdrawal }
    } else {
      const withdrawalTime = time - 1
      const {
        contributionToOrWithdrawalFromSavingsPortfolio:
          netContributionOrWithdrawal,
      } = simulate(withdrawalTime, estimate)
      const rebalanceTime = time + 1
      const { allocation } = simulate(rebalanceTime, estimate).afterWithdrawals
      return { allocation, netContributionOrWithdrawal }
    }
  }

  const start = portfolioBalance.isLastPlanChange
    ? portfolioBalance
    : portfolioBalance.original

  // -----------
  // Plan Change Stock Allocation Actions
  // -----------
  const planChangeAllocations = [
    ...(portfolioBalance.isLastPlanChange
      ? []
      : portfolioBalance.history.planChangeStockAllocations),
    {
      effectiveAtMarketCloseTime:
        marketData.dailyStockMarketPerformance[
          _getLastCompletedMarketDay(
            params.plan.timestamp,
            marketData.dailyStockMarketPerformance,
          )
        ].closingTime,
      allocation: (currentEstimate: number) =>
        simulate(params.plan.timestamp, currentEstimate).afterWithdrawals
          .allocation,
    },
  ]
  if (
    planChangeAllocations.length >= 2 &&
    _.nth(planChangeAllocations, -1)?.effectiveAtMarketCloseTime ===
      _.nth(planChangeAllocations, -2)?.effectiveAtMarketCloseTime
  ) {
    // Last plan change rebalance was for the same market close.
    planChangeAllocations.splice(planChangeAllocations.length - 2, 1)
  }
  const planChangeRebalanceActions = planChangeAllocations.map((x) => ({
    time: x.effectiveAtMarketCloseTime,
    getArgs: (currentEstimate: number) => ({
      type: 'planChangeRebalance' as const,
      allocation:
        typeof x.allocation === 'function'
          ? x.allocation(currentEstimate)
          : x.allocation,
    }),
  }))

  // -----------
  // Market Close Actions
  // -----------
  const marketCloseActions: _ActionSpec[] =
    marketData.dailyStockMarketPerformance
      .slice(
        _getLastCompletedMarketDay(
          start.timestamp,
          marketData.dailyStockMarketPerformance,
        ) + 1,
      )
      .map((x) => ({
        time: x.closingTime,
        getArgs: () => ({ type: 'marketClose', marketData: x }),
      }))

  // -----------
  // Month Boundary Actions
  // -----------
  const startOfMonthActions: _ActionSpec[] = _getStartOfMonthsBetween(
    getDateTimeInCurrentTimezone.fromMillis(start.timestamp),
    currentTime,
  ).map((startOfMonth) => {
    return {
      time: startOfMonth,
      getArgs: (currentEstimate) => {
        const { allocation, netContributionOrWithdrawal } =
          getAllocationAndWithdrawalForMonthBoundary(
            startOfMonth,
            currentEstimate,
          )
        return {
          type: 'monthBoundary',
          netContributionOrWithdrawal,
          allocation,
        }
      },
    }
  })

  let currState: _State = {
    estimate: start.amount,
    allocation: fGet(_.first(planChangeRebalanceActions)).getArgs(start.amount)
      .allocation,
  }

  let actions = [] as CurrentPortfolioBalanceEstimateAction[]
  while (
    planChangeRebalanceActions.length > 0 ||
    marketCloseActions.length > 0 ||
    startOfMonthActions.length > 0
  ) {
    actions.push(
      _applyAction(
        _pickNextAction([
          marketCloseActions,
          planChangeRebalanceActions,
          startOfMonthActions,
        ]),
        currState,
      ),
    )
    currState = fGet(_.last(actions)).stateChange.end
  }

  const result = {
    value: currState.estimate,
    entered: portfolioBalance.isLastPlanChange
      ? portfolioBalance
      : portfolioBalance.original,
    actions,
  }
  return result
}

const _applyAction = (
  {
    time,
    getArgs,
  }: {
    time: number
    getArgs: (
      startingEstimate: number,
    ) => CurrentPortfolioBalanceEstimateAction['args']
  },
  start: _State,
): CurrentPortfolioBalanceEstimateAction => {
  const args = getArgs(start.estimate)
  const end = ((): _State => {
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
      case 'monthBoundary':
        return {
          estimate:
            start.estimate +
            (args.netContributionOrWithdrawal.type === 'contribution'
              ? args.netContributionOrWithdrawal.contribution
              : -args.netContributionOrWithdrawal.withdrawal),
          allocation: args.allocation,
        }
      case 'planChangeRebalance':
        return {
          estimate: start.estimate,
          allocation: args.allocation,
        }
      default:
        noCase(args)
    }
  })()
  return { time, args, stateChange: { start, end } }
}

const _pickNextAction = <T extends { time: number }>(actionArrays: T[][]) => {
  let resultIndex = 0
  let minSortTime = Infinity
  actionArrays.forEach((actionArray, index) => {
    if (actionArray.length > 0 && actionArray[0].time < minSortTime) {
      minSortTime = actionArray[0].time
      resultIndex = index
    }
  })
  return fGet(actionArrays[resultIndex].shift())
}

const _getSimulate =
  (
    params: Params,
    marketData: MarketData,
    getDateTimeInCurrentTimezone: ReturnType<typeof getDateTimeInSameTimeZone>,
    wasm: WASM,
  ) =>
  (time: number, portfolioBalance: number) => {
    time = Math.max(time, params.plan.timestamp)
    const dateTime = getDateTimeInCurrentTimezone.fromMillis(time)
    const marketDataAtTime: MarketData['latest'] = {
      inflation: _getMarketDataAtTime(dateTime, marketData.inflation),
      CAPE: _getMarketDataAtTime(dateTime, marketData.CAPE),
      bondRates: _getMarketDataAtTime(dateTime, marketData.bondRates),
    }
    const paramsExt = extendParams(params, dateTime)
    const paramsProcessed = processPlanParams(
      paramsExt,
      portfolioBalance,
      marketDataAtTime,
    )

    const run = runSimulationInWASM(paramsProcessed, { start: 0, end: 1 }, wasm)
    return firstMonthSavingsPortfolioDetail(
      run.byMonthsFromNowByRun.savingsPortfolio,
      paramsProcessed,
    )
  }

const _getMarketDataAtTime = <T extends { closingTime: number }>(
  time: DateTime,
  data: T[],
): T => {
  const insertionPoint = _.sortedIndexBy(
    data,
    { closingTime: time.valueOf() },
    (x) => x.closingTime,
  )
  assert(insertionPoint > 0)
  return data[insertionPoint - 1]
}

// start is open, end in closed.
const _getStartOfMonthsBetween = (start: DateTime, end: DateTime) => {
  const firstMonth = start.startOf('month').plus({ month: 1 })
  const numMonths =
    fGet(end.startOf('month').diff(firstMonth, 'month').toObject().months) + 1
  return _.times(numMonths).map((i) => firstMonth.plus({ month: i }).valueOf())
}

const _getLastCompletedMarketDay = (
  time: number,
  dailyStockMarketPerformance: { closingTime: number }[],
) => {
  const insertionPoint = _.sortedIndexBy(
    dailyStockMarketPerformance,
    { closingTime: time },
    (x) => x.closingTime,
  )

  assert(insertionPoint > 0)
  return insertionPoint - 1
}

export const getHistoryForPortfolioBalanceEstimate = (
  { actions }: ReturnType<typeof estimateCurrentPortfolioBalance>,
  getDateTimeInCurrentTimezone: ReturnType<typeof getDateTimeInSameTimeZone>,
): Extract<
  PlanParams['wealth']['portfolioBalance'],
  { isLastPlanChange: false }
>['history'] => {
  const monthBoundaryDetails = actions
    .filter((x) => x.args.type === 'monthBoundary')
    .map((x) => {
      assert(x.args.type === 'monthBoundary')
      const { netContributionOrWithdrawal, allocation } = x.args
      const { year, month } = getDateTimeInCurrentTimezone.fromMillis(
        x.time + 1,
      )
      return {
        startOfMonth: { year, month },
        allocation,
        netContributionOrWithdrawal,
      }
    })

  const planChangeStockAllocations = actions
    .filter((x) => x.args.type === 'planChangeRebalance')
    .map((x) => {
      assert(x.args.type === 'planChangeRebalance')
      return {
        effectiveAtMarketCloseTime: x.time,
        allocation: x.args.allocation,
      }
    })
  return { monthBoundaryDetails, planChangeStockAllocations }
}
