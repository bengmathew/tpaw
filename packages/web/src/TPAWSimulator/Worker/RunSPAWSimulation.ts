import _ from 'lodash'
import {AccountForWithdrawal} from '../../Utils/AccountForWithdrawal'
import {assert} from '../../Utils/Utils'
import {TPAWParamsExt} from '../TPAWParamsExt'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'
import {simulationResult, SimulationResult} from './SimulationResult'

type _SingleYearResult = {
  savingsPortfolio: SavingsPortfolioThroughAYear.End
  wealthLessEssentialExpenses: number
}

export type SPAWSimulationResult = SimulationResult<_SingleYearResult>

export function runSPAWSimulation(
  params: TPAWParamsProcessed,
  paramsExt: TPAWParamsExt,
  args:
    | {
        type: 'useExpectedReturns'
      }
    | {
        type: 'useHistoricalReturns'
        resultsFromUsingExpectedReturns: SPAWSimulationResult
        randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
      }
): SPAWSimulationResult {
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
  const {asYFN, withdrawalStartYear} = paramsExt
  const withdrawalStarted = yearIndex >= asYFN(withdrawalStartYear)
  const {netPresentValue, cumulative1PlusGOver1PlusR} =
    params.preCalculations.forSPAW

  // ---- START -----
  const savingsPortfolioAtStart: SavingsPortfolioThroughAYear.Start = {
    start: {
      balance: prev
        ? prev.savingsPortfolio.end.balance
        : params.savingsAtStartOfStartYear,
    },
  }

  const wealthLessEssentialExpenses =
    savingsPortfolioAtStart.start.balance +
    netPresentValue.savings.withCurrentYear[yearIndex] -
    netPresentValue.withdrawals.essential.withCurrentYear[yearIndex]

  // ---- SCALE ----
  const scale = (() => {
    if (args.type === 'useExpectedReturns') return {discretionaryAndLegacy: 1}

    const discretionaryAndLegacy =
      wealthLessEssentialExpenses /
      args.resultsFromUsingExpectedReturns.wealthLessEssentialExpenses
    return {discretionaryAndLegacy}
  })()

  // ---- TARGET WITHDRAWALS ----
  const withdrawalTarget = (() => {
    const essential = params.byYear[yearIndex].withdrawals.essential
    let discretionary =
      params.byYear[yearIndex].withdrawals.discretionary *
      scale.discretionaryAndLegacy

    let regular = (() => {
      if (!withdrawalStarted) return 0
      return Math.max(
        (savingsPortfolioAtStart.start.balance +
          netPresentValue.savings.withCurrentYear[yearIndex] -
          netPresentValue.withdrawals.essential.withCurrentYear[yearIndex] -
          netPresentValue.withdrawals.discretionary.withCurrentYear[yearIndex] *
            scale.discretionaryAndLegacy -
          netPresentValue.legacy.withCurrentYear[yearIndex] *
            scale.discretionaryAndLegacy) /
          cumulative1PlusGOver1PlusR[yearIndex],
        0
      )
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
    const account = new AccountForWithdrawal(
      savingsPortfolioAfterContributions.afterContributions.balance
    )
    const essential = account.withdraw(withdrawalTarget.essential)
    const discretionary = account.withdraw(withdrawalTarget.discretionary)
    const regular = account.withdraw(withdrawalTarget.regular)
    return {discretionary, essential, regular}
  })()

  // ---- APPLY WITHDRAWALS ----
  const savingsPortfolioAfterWithdrawals =
    SavingsPortfolioThroughAYear.applyWithdrawals(
      withdrawals,
      savingsPortfolioAfterContributions
    )

  // ---- APPLY ASSET ALLOCATION --------

  const allocation = {
    stocks: params.targetAllocation.regularPortfolio.forSPAW[yearIndex],
  }
  const savingsPortfolioAtEnd = SavingsPortfolioThroughAYear.applyAllocation(
    allocation,
    args.type === 'useExpectedReturns'
      ? params.returns.expected
      : args.historicalReturn,
    savingsPortfolioAfterWithdrawals
  )

  return {
    savingsPortfolio: savingsPortfolioAtEnd,
    wealthLessEssentialExpenses,
  }
}
