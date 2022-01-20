import _ from 'lodash'
import { numOfYears } from '../Utils/NumOfYears'
import { fGet } from '../Utils/Utils'
import { TPAWParams } from './TPAWParams'
import { processTPAWParams, TPAWParamsProcessed } from './TPAWParamsProcessed'

export type TPAWSimulationResult = ReturnType<typeof runTPAWSimulation>
export function runTPAWSimulation(
  paramsIn: TPAWParams,
  opts: {
    randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
  } = {}
) {
  const params = processTPAWParams(
    paramsIn,
    opts.randomIndexesIntoHistoricalReturnsByYear
  )
  const byYearFromNow: TPAWSimulationForYear[] = []

  _.range(numOfYears(params.age)).reduce((prev, yearIndex) => {
    const result = runASingleYear(params, yearIndex, prev)
    byYearFromNow.push(result)
    return result
  }, null as null | TPAWSimulationForYear)
  return {
    byYearFromNow,
    legacy:
      fGet(_.last(byYearFromNow)).savingsPortfolioEndingBalance +
      params.legacy.external,
  }
}

export type TPAWSimulationForYear = ReturnType<typeof runASingleYear>
export function runASingleYear(
  params: TPAWParamsProcessed,
  yearIndex: number,
  prev: {
    returns: {
      expected: {legacyPortfolio: number; regularPortfolio: number}
      realized: {legacyPortfolio: number; regularPortfolio: number}
    }
    growthOfADollar: {
      expected: {legacyPortfolio: number; regularPortfolio: number}
      realized: {legacyPortfolio: number; regularPortfolio: number}
    }
    savingsPortfolioEndingBalance: number
  } | null
) {
  // ------ RETURNS -----
  const returns = (() => {
    const _expected = (allocation: {stocks: number; bonds: number}) =>
      params.returns.expected.bonds * allocation.bonds +
      params.returns.expected.stocks * allocation.stocks
    const _realized = (allocation: {stocks: number; bonds: number}) =>
      params.returns.realized[yearIndex].bonds * allocation.bonds +
      params.returns.realized[yearIndex].stocks * allocation.stocks

    return {
      expected: {
        stocks: params.returns.expected.stocks,
        bonds: params.returns.expected.bonds,
        essentialExpenses: _expected({stocks: 0, bonds: 1}),
        legacyPortfolio: _expected(params.targetAllocation.legacyPortfolio),
        regularPortfolio: _expected(params.targetAllocation.regularPortfolio),
      },
      realized: {
        stocks: params.returns.realized[yearIndex].stocks,
        bonds: params.returns.realized[yearIndex].bonds,
        essentialExpenses: _realized({stocks: 0, bonds: 1}),
        legacyPortfolio: _realized(params.targetAllocation.legacyPortfolio),
        regularPortfolio: _realized(params.targetAllocation.regularPortfolio),
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

  // ---- GROWTH OF A DOLLAR ----
  const growthOfADollar = (() => {
    return {
      expected: {
        legacyPortfolio: prev
          ? prev.growthOfADollar.expected.legacyPortfolio *
            (1 + prev.returns.expected.legacyPortfolio)
          : 1,

        regularPortfolio: prev
          ? prev.growthOfADollar.expected.regularPortfolio *
            (1 + prev.returns.expected.regularPortfolio)
          : 1,
      },
      realized: {
        legacyPortfolio: prev
          ? prev.growthOfADollar.realized.legacyPortfolio *
            (1 + prev.returns.realized.legacyPortfolio)
          : 1,

        regularPortfolio: prev
          ? prev.growthOfADollar.realized.regularPortfolio *
            (1 + prev.returns.realized.regularPortfolio)
          : 1,
      },
    }
  })()

  // ---- SCALE ----
  const scale = (() => ({
    legacyPortfolio:
      growthOfADollar.realized.legacyPortfolio /
      growthOfADollar.expected.legacyPortfolio,
    regularPortfolio:
      growthOfADollar.realized.regularPortfolio /
      growthOfADollar.expected.regularPortfolio,
  }))()

  // ---- WEALTH AND SPENDING ----
  const wealthAndSpending = (() => {
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
    const presentValueOfExtraWithdrawals = Math.min(
      netPresentValue.withdrawals.fundedByRiskPortfolio('addCurrYear') *
        scale.regularPortfolio,
      wealth - presentValueOfEssentialExpenses
    )
    const presentValueOfDesiredLegacy = Math.min(
      (params.legacy.target * scale.legacyPortfolio) /
        Math.pow(
          1 + returns.expected.legacyPortfolio,
          numOfYears(params.age) - yearIndex
        ),
      wealth - presentValueOfEssentialExpenses - presentValueOfExtraWithdrawals
    )
    const presentValuesOfRegularWithdrawals =
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
      presentValuesOfRegularWithdrawals,
    }
  })()

  // ---- WITHDRAWAL TARGET ----
  const withdrawalTarget = (() => {
    const essential = params.byYear[yearIndex].withdrawals.fundedByBonds
    const legacy = 0
    let extra =
      params.byYear[yearIndex].withdrawals.fundedByRiskPortfolio *
      scale.regularPortfolio

    let regular = (() => {
      if (params.age.start + yearIndex < params.age.retirement) return 0

      const P = wealthAndSpending.presentValuesOfRegularWithdrawals
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

    const essentail = Math.min(withdrawalTarget.essential, availableFunds)
    const legacy = 0
    const extra = Math.min(withdrawalTarget.extra, availableFunds - essentail)
    const regular =
      params.age.start + yearIndex < params.age.retirement
        ? 0
        : Math.min(withdrawalTarget.regular, availableFunds - essentail - extra)
    const total = essentail + legacy + extra + regular
    const fromSavings = total - currentYearIncome

    return {
      savings,
      currentYearIncome,
      availableFunds,
      essentail,
      legacy,
      extra,
      regular,
      total,
      fromSavings,
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
        scale.regularPortfolio,
      wealth - presentValueOfEssentialExpenses
    )
    const presentValueOfDesiredLegacy = Math.min(
      (params.legacy.target * scale.legacyPortfolio) /
        Math.pow(
          1 + returns.expected.legacyPortfolio,
          numOfYears(params.age) - yearIndex
        ),
      wealth - presentValueOfEssentialExpenses - presentValueOfExtraWithdrawals
    )
    const presentValuesOfRegularWithdrawals =
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
      presentValuesOfRegularWithdrawals,
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
      portfolioAfterIncomeAndWithdrawals.presentValuesOfRegularWithdrawals *
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

  // ---- WITHDRAWAL ----
  const withdrawal = withdrawalAchieved.total

  return {
    returns,
    scale,
    growthOfADollar,
    wealthAndSpending,
    withdrawalTarget,
    withdrawalAchieved,
    portfolioAfterIncomeAndWithdrawals,
    savingsPortfolioAllocation,
    savingsPortfolioEndingBalance,
    withdrawal,
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
