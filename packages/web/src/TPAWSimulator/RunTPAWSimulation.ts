import _ from 'lodash'
import { numOfYears } from '../Utils/NumOfYears'
import { assert, fGet } from '../Utils/Utils'
import { TPAWParams } from './TPAWParams'
import { processTPAWParams, TPAWParamsProcessed } from './TPAWParamsProcessed'

export type TPAWSimulationResult = ReturnType<typeof runTPAWSimulation>
export function runTPAWSimulation(
  paramsIn: TPAWParams,
  realizedReturnsFromSimulation: {
    simulationUsingExpectedReturns: TPAWSimulationForYear[]
    randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
  } | null
) {
  const params = processTPAWParams(
    paramsIn,
    realizedReturnsFromSimulation
      ? realizedReturnsFromSimulation.randomIndexesIntoHistoricalReturnsByYear
      : undefined
  )
  if (realizedReturnsFromSimulation)
    assert(
      realizedReturnsFromSimulation.simulationUsingExpectedReturns.length ===
        numOfYears(params.age)
    )

  const byYearFromNow: TPAWSimulationForYear[] = []

  _.range(numOfYears(params.age)).reduce((prev, yearIndex) => {
    const result = runASingleYear(
      params,
      yearIndex,
      realizedReturnsFromSimulation
        ? {
            simulationUsingExpectedReturns:
              realizedReturnsFromSimulation.simulationUsingExpectedReturns[
                yearIndex
              ],
          }
        : null,
      prev
    )
    byYearFromNow.push(result)
    return result
  }, null as null | TPAWSimulationForYear)

  const lastYear = fGet(_.last(byYearFromNow))
  const endingBalanceOfSavingsPortfolio = lastYear.savingsPortfolioEndingBalance
  const legacy = endingBalanceOfSavingsPortfolio + params.legacy.external
  return {byYearFromNow, endingBalanceOfSavingsPortfolio, legacy}
}

export type TPAWSimulationForYear = ReturnType<typeof runASingleYear>
export function runASingleYear(
  params: TPAWParamsProcessed,
  yearIndex: number,
  realizedReturnsFromSimulation: {
    simulationUsingExpectedReturns: {
      wealthAndSpending: {
        wealth: number
        presentValueOfExtraWithdrawals: number
        presentValueOfDesiredLegacy: number
        presentValueOfRegularWithdrawals: number
      }
    }
  } | null,
  prev: {
    savingsPortfolioEndingBalance: number
  } | null
) {
  // ------ RETURNS -----
  const returns = (() => {
    const _expected = (allocation: {stocks: number; bonds: number}) =>
      params.returns.expected.bonds * allocation.bonds +
      params.returns.expected.stocks * allocation.stocks
    const _realized = realizedReturnsFromSimulation
      ? (allocation: {stocks: number; bonds: number}) =>
          params.returns.realized[yearIndex].bonds * allocation.bonds +
          params.returns.realized[yearIndex].stocks * allocation.stocks
      : _expected

    return {
      expected: {
        stocks: _expected({stocks: 1, bonds: 0}),
        bonds: _expected({stocks: 0, bonds: 1}),
        essentialExpenses: _expected({stocks: 0, bonds: 1}),
        legacyPortfolio: _expected(params.targetAllocation.legacyPortfolio),
        regularPortfolio: _expected(params.targetAllocation.regularPortfolio),
      },
      realized: {
        stocks: _realized({stocks: 1, bonds: 0}),
        bonds: _realized({stocks: 0, bonds: 1}),
      },
    }
  })()

  // ---- NET PRESENT VALUE ----
  const netPresentValue = (() => {
    const withOpt =
      (value: number, curr: number) =>
      (addCurrYear: 'addCurrYear' | 'skipCurrYear') =>
        value + (addCurrYear === 'addCurrYear' ? curr : 0)

    const paramsForCurrYear = params.byYear[yearIndex]
    const paramsForRemainingYears = params.byYear.slice(yearIndex + 1)

    const savings = withOpt(
      getNetPresentValue(
        returns.expected.bonds,
        paramsForRemainingYears.map(x => x.savings)
      ),
      paramsForCurrYear.savings
    )

    const withdrawals = {
      fundedByBonds: withOpt(
        getNetPresentValue(
          returns.expected.bonds,
          paramsForRemainingYears.map(x => x.withdrawals.fundedByBonds)
        ),
        paramsForCurrYear.withdrawals.fundedByBonds
      ),
      fundedByRiskPortfolio: withOpt(
        getNetPresentValue(
          returns.expected.regularPortfolio,
          paramsForRemainingYears.map(x => x.withdrawals.fundedByRiskPortfolio)
        ),
        paramsForCurrYear.withdrawals.fundedByRiskPortfolio
      ),
    }
    return {savings, withdrawals}
  })()

  // ---- WEALTH AND SPENDING ----
  const {scale, ...wealthAndSpending} = (() => {
    const startingBalanceOfSavingsPortfolio = prev
      ? prev.savingsPortfolioEndingBalance
      : params.savingsAtStartOfStartYear
    const presentValueOfFutureSavings = netPresentValue.savings('addCurrYear')

    const wealth =
      startingBalanceOfSavingsPortfolio + presentValueOfFutureSavings
    const presentValueOfEssentialExpenses = Math.min(
      netPresentValue.withdrawals.fundedByBonds('addCurrYear'),
      wealth
    )

    const scale = (() => {
      if (!realizedReturnsFromSimulation) return {legacy: 0, extraWithdrawls: 0}
      const {simulationUsingExpectedReturns} = realizedReturnsFromSimulation

      const elasticityOfWealthWRTStocks =
        simulationUsingExpectedReturns.wealthAndSpending.wealth === 0
          ? (params.targetAllocation.legacyPortfolio.stocks +
              params.targetAllocation.regularPortfolio.stocks +
              params.targetAllocation.regularPortfolio.stocks) /
            3
          : (simulationUsingExpectedReturns.wealthAndSpending
              .presentValueOfDesiredLegacy /
              simulationUsingExpectedReturns.wealthAndSpending.wealth) *
              params.targetAllocation.legacyPortfolio.stocks +
            (simulationUsingExpectedReturns.wealthAndSpending
              .presentValueOfExtraWithdrawals /
              simulationUsingExpectedReturns.wealthAndSpending.wealth) *
              params.targetAllocation.regularPortfolio.stocks +
            (simulationUsingExpectedReturns.wealthAndSpending
              .presentValueOfRegularWithdrawals /
              simulationUsingExpectedReturns.wealthAndSpending.wealth) *
              params.targetAllocation.regularPortfolio.stocks

      const elasticityOfExtraWithdrawalGoalsWRTWealth =
        elasticityOfWealthWRTStocks === 0
          ? 0
          : params.targetAllocation.regularPortfolio.stocks /
            elasticityOfWealthWRTStocks

      const elasticityOfLegacyGoalsWRTWealth =
        elasticityOfWealthWRTStocks === 0
          ? 0
          : params.targetAllocation.legacyPortfolio.stocks /
            elasticityOfWealthWRTStocks

      const percentIncreaseInWealthOverScheduled =
        simulationUsingExpectedReturns.wealthAndSpending.wealth === 0
          ? 0
          : wealth / simulationUsingExpectedReturns.wealthAndSpending.wealth - 1

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

    const presentValueOfExtraWithdrawals = Math.min(
      netPresentValue.withdrawals.fundedByRiskPortfolio('addCurrYear') *
        (1 + scale.extraWithdrawls),
      wealth - presentValueOfEssentialExpenses
    )
    const presentValueOfDesiredLegacy = Math.min(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(
          1 + returns.expected.legacyPortfolio,
          numOfYears(params.age) - yearIndex
        ),
      wealth - presentValueOfEssentialExpenses - presentValueOfExtraWithdrawals
    )
    const presentValueOfRegularWithdrawals =
      wealth -
      presentValueOfEssentialExpenses -
      presentValueOfDesiredLegacy -
      presentValueOfExtraWithdrawals

    return {
      startingBalanceOfSavingsPortfolio,
      presentValueOfFutureSavings,
      wealth,
      presentValueOfEssentialExpenses,
      presentValueOfExtraWithdrawals,
      presentValueOfDesiredLegacy,
      presentValueOfRegularWithdrawals,
      scale,
    }
  })()

  // ---- WITHDRAWAL TARGET ----
  const withdrawalTarget = (() => {
    const essential = params.byYear[yearIndex].withdrawals.fundedByBonds
    const legacy = 0
    let extra =
      params.byYear[yearIndex].withdrawals.fundedByRiskPortfolio *
      (1 + scale.extraWithdrawls)

    let regular = (() => {
      if (params.age.start + yearIndex < params.age.retirement) return 0

      const P = wealthAndSpending.presentValueOfRegularWithdrawals
      const r = returns.expected.regularPortfolio
      const g = params.scheduledWithdrawalGrowthRate
      const n = numOfYears(params.age) - yearIndex
      if (Math.abs(r - g) < 0.0000000001) return P / n
      return (P * (r - g)) / ((1 - Math.pow((1 + g) / (1 + r), n)) * (1 + r))
    })()

    if (params.spendingCeiling !== null) {
      extra = Math.min(
        extra,
        params.byYear[yearIndex].withdrawals.fundedByRiskPortfolio
      )
      regular = Math.min(regular, params.spendingCeiling)
    }
    if (params.spendingFloor !== null) {
      extra = Math.max(
        extra,
        params.byYear[yearIndex].withdrawals.fundedByRiskPortfolio
      )
      regular = Math.max(regular, params.spendingFloor)
    }
    return {essential, legacy, extra, regular}
  })()

  // ---- WITHDRAWAL ACHIEVED ----
  const withdrawalAchieved = (() => {
    const savings = wealthAndSpending.startingBalanceOfSavingsPortfolio
    const currentYearIncome = params.byYear[yearIndex].savings
    const availableFunds = savings + currentYearIncome

    const essential = Math.min(withdrawalTarget.essential, availableFunds)
    const legacy = 0
    const extra = Math.min(withdrawalTarget.extra, availableFunds - essential)
    const regular =
      params.age.start + yearIndex < params.age.retirement
        ? 0
        : Math.min(withdrawalTarget.regular, availableFunds - essential - extra)
    const total = essential + legacy + extra + regular
    const fromSavings = total - currentYearIncome
    const fromSavingsRate = savings === 0 ? 0 : fromSavings / savings

    return {
      savings,
      currentYearIncome,
      availableFunds,
      essential,
      legacy,
      extra,
      regular,
      total,
      fromSavings,
      fromSavingsRate,
    }
  })()

  // ---- PORTFOLIO AFTER INCOME AND WITHDRAWALS ----
  const portfolioAfterIncomeAndWithdrawals = (() => {
    const savings = withdrawalAchieved.availableFunds - withdrawalAchieved.total
    const presentValueOfFutureSavings = netPresentValue.savings('skipCurrYear')
    const wealth = savings + presentValueOfFutureSavings
    const presentValueOfEssentialExpenses = Math.min(
      netPresentValue.withdrawals.fundedByBonds('skipCurrYear'),
      wealth
    )
    const presentValueOfExtraWithdrawals = Math.min(
      netPresentValue.withdrawals.fundedByRiskPortfolio('skipCurrYear') *
        (1 + scale.extraWithdrawls),
      wealth - presentValueOfEssentialExpenses
    )
    const presentValueOfDesiredLegacy = Math.min(
      (params.legacy.target * (1 + scale.legacy)) /
        Math.pow(
          1 + returns.expected.legacyPortfolio,
          numOfYears(params.age) - yearIndex
        ),
      wealth - presentValueOfEssentialExpenses - presentValueOfExtraWithdrawals
    )
    const presentValueOfRegularWithdrawals =
      wealth -
      presentValueOfEssentialExpenses -
      presentValueOfDesiredLegacy -
      presentValueOfExtraWithdrawals
    return {
      savings,
      presentValueOfFutureSavings,
      wealth,
      presentValueOfEssentialExpenses,
      presentValueOfDesiredLegacy,
      presentValueOfExtraWithdrawals,
      presentValueOfRegularWithdrawals,
    }
  })()

  // ---- SAVINGS PORTFOLIO ALLOCATION ----
  const savingsPortfolioAllocation = (() => {
    const balance = portfolioAfterIncomeAndWithdrawals.savings
    const stocksTarget =
      portfolioAfterIncomeAndWithdrawals.presentValueOfDesiredLegacy *
        params.targetAllocation.legacyPortfolio.stocks +
      portfolioAfterIncomeAndWithdrawals.presentValueOfExtraWithdrawals *
        params.targetAllocation.regularPortfolio.stocks +
      portfolioAfterIncomeAndWithdrawals.presentValueOfRegularWithdrawals *
        params.targetAllocation.regularPortfolio.stocks

    const stocksAchieved = Math.min(balance, stocksTarget)
    const bonds = balance - stocksAchieved
    const asPercentage = {
      stocks: balance > 0 ? stocksAchieved / balance : null,
      bonds: balance > 0 ? bonds / balance : null,
    }

    return {balance, stocksTarget, stocksAchieved, bonds, asPercentage}
  })()

  // ---- SAVINGS PORTFOLIO ENDING BALANCE ----
  const savingsPortfolioEndingBalance =
    savingsPortfolioAllocation.stocksAchieved * (1 + returns.realized.stocks) +
    savingsPortfolioAllocation.bonds * (1 + returns.realized.bonds)

  return {
    returns,
    scale,
    wealthAndSpending,
    withdrawalTarget,
    withdrawalAchieved,
    portfolioAfterIncomeAndWithdrawals,
    savingsPortfolioAllocation,
    savingsPortfolioEndingBalance,
  }
}

// This is where most of the computational cost lies.
export const getNetPresentValue = (
  rate: number,
  amounts: number[],
  timesOnePlusRate?: 'timesOnePlusRate'
) =>
  // Significantly faster than Math.pow(), _.sum, and _.map.
  amounts.reduce(
    (p, amount) => {
      const newRate = p.rate * (1 + rate)
      return {rate: newRate, sum: p.sum + amount / newRate}
    },
    {rate: 1, sum: 0}
  ).sum * (timesOnePlusRate ? 1 + rate : 1)
