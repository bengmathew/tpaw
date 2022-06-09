import _ from 'lodash'
import {AccountForWithdrawal} from '../../Utils/AccountForWithdrawal'
import {blendReturns} from '../../Utils/BlendReturns'
import {assert} from '../../Utils/Utils'
import {getNumYears, getWithdrawalStartAsYFN, TPAWParamsExt} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'
import {simulationResult, SimulationResult} from './SimulationResult'

type _SingleYearResult = {
  savingsPortfolio: SavingsPortfolioThroughAYear.End
  wealth: number
  presentValueOfSpending: {
    withdrawals: {
      regular: number
      discretionary: number
    }
    legacy: number
  }
}
export type TPAWSimulationResult = SimulationResult<_SingleYearResult>

export function runTPAWSimulation(
  params: TPAWParamsProcessed,
  args:
    | {type: 'useExpectedReturns'}
    | {
        type: 'useHistoricalReturns'
        resultsFromUsingExpectedReturns: TPAWSimulationResult
        randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
      }
): TPAWSimulationResult {
  const numYears = getNumYears(params.original)
  if (args.type === 'useHistoricalReturns')
    assert(
      args.resultsFromUsingExpectedReturns.byYearFromNow.length === numYears
    )

  const {historicalAdjusted} = params.returns
  const byYearFromNow: _SingleYearResult[] = []

  _.range(numYears).reduce((prev, yearIndex) => {
    const result = runASingleYear(
      params,
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

  const numYears = getNumYears(params.original)
  const withdrawalStartAsYFN = getWithdrawalStartAsYFN(params.original)
  const withdrawalStarted = yearIndex >= withdrawalStartAsYFN

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
    const account = new AccountForWithdrawal(wealth)
    account.withdraw(netPresentValue.withdrawals.lmp.withCurrentYear[yearIndex])
    account.withdraw(
      netPresentValue.withdrawals.essential.withCurrentYear[yearIndex]
    )
    const discretionary = account.withdraw(
      netPresentValue.withdrawals.discretionary.withCurrentYear[yearIndex] *
        (1 + scale.extraWithdrawls)
    )
    const legacy = account.withdraw(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(1 + expectedReturns.legacyPortfolio, numYears - yearIndex)
    )
    const regular = account.balance

    return {withdrawals: {regular, discretionary}, legacy}
  })()

  // ---- WITHDRAWAL TARGET ----
  const withdrawalTarget = (() => {
    const lmp = !withdrawalStarted ? 0 : params.withdrawals.lmp
    const essential = params.byYear[yearIndex].withdrawals.essential
    let discretionary =
      params.byYear[yearIndex].withdrawals.discretionary *
      (1 + scale.extraWithdrawls)

    let regularWithLMP =
      lmp +
      (() => {
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
      regularWithLMP = Math.min(regularWithLMP, params.spendingCeiling)
    }
    if (params.spendingFloor !== null) {
      discretionary = Math.max(
        discretionary,
        params.byYear[yearIndex].withdrawals.discretionary
      )
      regularWithLMP = withdrawalStarted
        ? Math.max(regularWithLMP, params.spendingFloor)
        : regularWithLMP
    }
    const regularWithoutLMP = regularWithLMP - lmp
    assert(regularWithoutLMP >= 0)
    return {lmp, essential, discretionary, regularWithoutLMP}
  })()

  // ---- APPLY CONTRIBUTIONS ----
  const savingsPortfolioAfterContributions =
    SavingsPortfolioThroughAYear.applyContributions(
      params.byYear[yearIndex].savings,
      savingsPortfolioAtStart
    )

  // ---- WITHDRAWAL ACHIEVED ----
  const withdrawals = (() => {
    const account = new AccountForWithdrawal(
      savingsPortfolioAfterContributions.afterContributions.balance
    )
    const lmp = account.withdraw(withdrawalTarget.lmp)
    const essential = account.withdraw(withdrawalTarget.essential)
    const discretionary = account.withdraw(withdrawalTarget.discretionary)
    const regular = lmp + account.withdraw(withdrawalTarget.regularWithoutLMP)
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
    const account = new AccountForWithdrawal(
      savingsPortfolioAfterWithdrawals.afterWithdrawals.balance +
        netPresentValue.savings.withoutCurrentYear[yearIndex]
    )

    account.withdraw(
      netPresentValue.withdrawals.lmp.withoutCurrentYear[yearIndex]
    )
    account.withdraw(
      netPresentValue.withdrawals.essential.withoutCurrentYear[yearIndex]
    )

    const presentValueOfDiscretionaryWithdrawals = account.withdraw(
      netPresentValue.withdrawals.discretionary.withoutCurrentYear[yearIndex] *
        (1 + scale.extraWithdrawls)
    )
    const presentValueOfDesiredLegacy = account.withdraw(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(1 + expectedReturns.legacyPortfolio, numYears - yearIndex)
    )
    const presentValueOfRegularWithdrawals = account.balance

    const stocksTarget =
      presentValueOfDesiredLegacy *
        params.targetAllocation.legacyPortfolio.stocks +
      presentValueOfDiscretionaryWithdrawals *
        params.targetAllocation.regularPortfolio.forTPAW.stocks +
      presentValueOfRegularWithdrawals *
        params.targetAllocation.regularPortfolio.forTPAW.stocks

    const stocksAchieved = Math.min(
      savingsPortfolioAfterWithdrawals.afterWithdrawals.balance,
      stocksTarget
    )
    const allocation = {
      stocks:
        savingsPortfolioAfterWithdrawals.afterWithdrawals.balance > 0
          ? stocksAchieved /
            savingsPortfolioAfterWithdrawals.afterWithdrawals.balance
          : 0,
    }
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