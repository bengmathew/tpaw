import _ from 'lodash'
import {blendReturns} from '../../Utils/BlendReturns'
import {assert} from '../../Utils/Utils'
import {TPAWParamsExt} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'
import {simulationResult, SimulationResult} from './SimulationResult'

type _SingleYearResult = {
  savingsPortfolio: SavingsPortfolioThroughAYear.End
  wealth: number
  presentValueOfSpending: {
    withdrawals: {
      regular: number
      essential: number
      discretionary: number
    }
    legacy: number
  }
}
export type TPAWSimulationResult = SimulationResult<_SingleYearResult>

export function runTPAWSimulation(
  params: TPAWParamsProcessed,
  paramsExt: TPAWParamsExt,
  args:
    | {
        type: 'useExpectedReturns'
      }
    | {
        type: 'useHistoricalReturns'
        resultsFromUsingExpectedReturns: TPAWSimulationResult
        randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
      }
): TPAWSimulationResult {
  const {numYears} = paramsExt
  if (args.type === 'useHistoricalReturns')
    assert(
      args.resultsFromUsingExpectedReturns.byYearFromNow.length === numYears
    )

  const {historicalAdjusted} = params.returns
  const byYearFromNow: _SingleYearResult[] = []

  _.range(numYears).reduce((prev, yearIndex) => {
    const result = runASingleYear(
      params,
      paramsExt,
      yearIndex,
      args.type === 'useExpectedReturns'
        ? {type: 'useExpectedReturns'}
        : {
            type: 'useHistoricalReturns',
            resultsFromUsingExpectedReturns:
              args.resultsFromUsingExpectedReturns.byYearFromNow[yearIndex],
            historicalReturn:
              historicalAdjusted[
                args.randomIndexesIntoHistoricalReturnsByYear?.(yearIndex) ??
                  _.random(historicalAdjusted.length - 1)
              ],
          },
      prev
    )
    byYearFromNow.push(result)
    return result
  }, null as null | _SingleYearResult)

  return simulationResult(byYearFromNow, params)
}

// ---------------------------
// ---- RUN A SINGLE YEAR ----
// ---------------------------

export function runASingleYear(
  params: TPAWParamsProcessed,
  paramsExt: TPAWParamsExt,
  yearIndex: number,
  args:
    | {type: 'useExpectedReturns'}
    | {
        type: 'useHistoricalReturns'
        historicalReturn: {stocks: number; bonds: number}
        resultsFromUsingExpectedReturns: _SingleYearResult
      },
  prev: _SingleYearResult | null
): _SingleYearResult {
  const {netPresentValue} = params.preCalculations.forTPAW

  const {asYFN, numYears, withdrawalStartYear} = paramsExt

  const withdrawalStarted = yearIndex >= asYFN(withdrawalStartYear)

  // ---- START SAVINGS PORTFOLIO ----
  const savingsPortfolioAtStart: SavingsPortfolioThroughAYear.Start = {
    start: {
      balance: prev
        ? prev.savingsPortfolio.end.balance
        : params.savingsAtStartOfStartYear,
    },
  }

  // ---- WEALTH ----
  const wealth =
    savingsPortfolioAtStart.start.balance +
    netPresentValue.savings.withCurrentYear[yearIndex]

  // ---- SCALE ----
  const scale = (() => {
    if (args.type === 'useExpectedReturns')
      return {legacy: 0, extraWithdrawls: 0}

    const elasticityOfWealthWRTStocks =
      args.resultsFromUsingExpectedReturns.wealth === 0
        ? (params.targetAllocation.legacyPortfolio.stocks +
            params.targetAllocation.regularPortfolio.forTPAW.stocks +
            params.targetAllocation.regularPortfolio.forTPAW.stocks) /
          3
        : (args.resultsFromUsingExpectedReturns.presentValueOfSpending.legacy /
            args.resultsFromUsingExpectedReturns.wealth) *
            params.targetAllocation.legacyPortfolio.stocks +
          (args.resultsFromUsingExpectedReturns.presentValueOfSpending
            .withdrawals.discretionary /
            args.resultsFromUsingExpectedReturns.wealth) *
            params.targetAllocation.regularPortfolio.forTPAW.stocks +
          (args.resultsFromUsingExpectedReturns.presentValueOfSpending
            .withdrawals.regular /
            args.resultsFromUsingExpectedReturns.wealth) *
            params.targetAllocation.regularPortfolio.forTPAW.stocks

    const elasticityOfExtraWithdrawalGoalsWRTWealth =
      elasticityOfWealthWRTStocks === 0
        ? 0
        : params.targetAllocation.regularPortfolio.forTPAW.stocks /
          elasticityOfWealthWRTStocks

    const elasticityOfLegacyGoalsWRTWealth =
      elasticityOfWealthWRTStocks === 0
        ? 0
        : params.targetAllocation.legacyPortfolio.stocks /
          elasticityOfWealthWRTStocks

    const percentIncreaseInWealthOverScheduled =
      args.resultsFromUsingExpectedReturns.wealth === 0
        ? 0
        : wealth / args.resultsFromUsingExpectedReturns.wealth - 1

    const legacy = Math.max(
      percentIncreaseInWealthOverScheduled * elasticityOfLegacyGoalsWRTWealth,
      -1
    )
    const extraWithdrawls = Math.max(
      percentIncreaseInWealthOverScheduled *
        elasticityOfExtraWithdrawalGoalsWRTWealth,
      -1
    )
    return {legacy, extraWithdrawls}
  })()

  // ------ RETURNS -----
  const expectedReturns = (() => {
    const _expected = blendReturns(params.returns.expected)
    return {
      legacyPortfolio: _expected(params.targetAllocation.legacyPortfolio),
      regularPortfolio: _expected(
        params.targetAllocation.regularPortfolio.forTPAW
      ),
    }
  })()

  // ---- PRESENT VALUE OF SPENDING ----
  const presentValueOfSpending = (() => {
    const essentialWithdrawals = Math.min(
      netPresentValue.withdrawals.essential.withCurrentYear[yearIndex],
      wealth
    )

    const discretionaryWithdrawals = Math.min(
      netPresentValue.withdrawals.discretionary.withCurrentYear[yearIndex] *
        (1 + scale.extraWithdrawls),
      wealth - essentialWithdrawals
    )
    const legacy = Math.min(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(1 + expectedReturns.legacyPortfolio, numYears - yearIndex),
      wealth - essentialWithdrawals - discretionaryWithdrawals
    )
    const regular =
      wealth - essentialWithdrawals - legacy - discretionaryWithdrawals

    return {
      withdrawals: {
        regular,
        essential: essentialWithdrawals,
        discretionary: discretionaryWithdrawals,
      },
      legacy,
    }
  })()

  // ---- WITHDRAWAL TARGET ----
  const withdrawalTarget = (() => {
    const essential = params.byYear[yearIndex].withdrawals.essential
    let discretionary =
      params.byYear[yearIndex].withdrawals.discretionary *
      (1 + scale.extraWithdrawls)

    let regular = (() => {
      if (!withdrawalStarted) return 0

      const P = presentValueOfSpending.withdrawals.regular
      const r = expectedReturns.regularPortfolio
      const g = params.scheduledWithdrawalGrowthRate
      const n = numYears - yearIndex
      if (Math.abs(r - g) < 0.0000000001) return P / n
      return (P * (r - g)) / ((1 - Math.pow((1 + g) / (1 + r), n)) * (1 + r))
    })()

    if (params.spendingCeiling !== null) {
      discretionary = Math.min(
        discretionary,
        params.byYear[yearIndex].withdrawals.discretionary
      )
      regular = Math.min(regular, params.spendingCeiling)
    }
    if (params.spendingFloor !== null) {
      discretionary = Math.max(
        discretionary,
        params.byYear[yearIndex].withdrawals.discretionary
      )
      regular = withdrawalStarted
        ? Math.max(regular, params.spendingFloor)
        : regular
    }
    return {essential, discretionary, regular}
  })()

  // ---- APPLY CONTRIBUTIONS ----
  const savingsPortfolioAfterContributions =
    SavingsPortfolioThroughAYear.applyContributions(
      params.byYear[yearIndex].savings,
      savingsPortfolioAtStart
    )

  // ---- WITHDRAWAL ACHIEVED ----
  const withdrawals = (() => {
    const essential = Math.min(
      withdrawalTarget.essential,
      savingsPortfolioAfterContributions.afterContributions.balance
    )
    const discretionary = Math.min(
      withdrawalTarget.discretionary,
      savingsPortfolioAfterContributions.afterContributions.balance - essential
    )
    const regular =
      yearIndex < asYFN(withdrawalStartYear)
        ? 0
        : Math.min(
            withdrawalTarget.regular,
            savingsPortfolioAfterContributions.afterContributions.balance -
              essential -
              discretionary
          )
    return {discretionary, essential, regular}
  })()

  // ---- APPLY WITHDRAWALS ----
  const savingsPortfolioAfterWithdrawals =
    SavingsPortfolioThroughAYear.applyWithdrawals(
      withdrawals,
      savingsPortfolioAfterContributions
    )

  // ---- CALCULATE ALLOCATION ----
  const {allocation} = (() => {
    const presentValueOfFutureSavings =
      netPresentValue.savings.withoutCurrentYear[yearIndex]
    const wealth =
      savingsPortfolioAfterWithdrawals.afterWithdrawals.balance +
      presentValueOfFutureSavings
    const presentValueOfEssentialWithdrawals = Math.min(
      netPresentValue.withdrawals.essential.withoutCurrentYear[yearIndex],
      wealth
    )
    const presentValueOfDiscretionaryWithdrawals = Math.min(
      netPresentValue.withdrawals.discretionary.withoutCurrentYear[yearIndex] *
        (1 + scale.extraWithdrawls),
      wealth - presentValueOfEssentialWithdrawals
    )
    const presentValueOfDesiredLegacy = Math.min(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(1 + expectedReturns.legacyPortfolio, numYears - yearIndex),
      wealth -
        presentValueOfEssentialWithdrawals -
        presentValueOfDiscretionaryWithdrawals
    )
    const presentValueOfRegularWithdrawals =
      wealth -
      presentValueOfEssentialWithdrawals -
      presentValueOfDesiredLegacy -
      presentValueOfDiscretionaryWithdrawals

    const balance = savingsPortfolioAfterWithdrawals.afterWithdrawals.balance
    const stocksTarget =
      presentValueOfDesiredLegacy *
        params.targetAllocation.legacyPortfolio.stocks +
      presentValueOfDiscretionaryWithdrawals *
        params.targetAllocation.regularPortfolio.forTPAW.stocks +
      presentValueOfRegularWithdrawals *
        params.targetAllocation.regularPortfolio.forTPAW.stocks

    const stocksAchieved = Math.min(balance, stocksTarget)
    const allocation = {stocks: balance > 0 ? stocksAchieved / balance : 0}
    return {allocation}
  })()

  // ---- APPLY ALLOCATION ----
  const savingsPortfolioAtEnd = SavingsPortfolioThroughAYear.applyAllocation(
    allocation,
    args.type === 'useExpectedReturns'
      ? params.returns.expected
      : args.historicalReturn,
    savingsPortfolioAfterWithdrawals
  )

  return {
    savingsPortfolio: savingsPortfolioAtEnd,
    wealth,
    presentValueOfSpending,
  }
}
