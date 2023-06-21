import _ from 'lodash'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
import { TPAWWorkerRunSimulationResult } from './TPAWWorkerAPI'

export type FirstMonthSavingsPortfolioDetail = {
  start: { balance: number }
  contributions: {
    total: number
    toWithdrawal: number
    toSavingsPortfolio: number
  }
  afterContributions: {
    balance: number
  }
  withdrawals: {
    regular: number
    essential: number
    discretionary: number
    total: number
    fromSavingsPortfolio: number
    fromContributions: number
  }
  afterWithdrawals: {
    allocation: { stocks: number }
    balance: number
  }
  contributionToOrWithdrawalFromSavingsPortfolio:
    | { type: 'contribution'; contribution: number }
    | { type: 'withdrawal'; withdrawal: number }
}

export const firstMonthSavingsPortfolioDetail = (
  byMonthsFromNowByRun: TPAWWorkerRunSimulationResult['byMonthsFromNowByRun']['savingsPortfolio'],
  params: PlanParamsProcessed,
): FirstMonthSavingsPortfolioDetail => {
  const start = {
    balance: byMonthsFromNowByRun.start.balance[0][0],
  }
  const contributionsTotal =
    params.byMonth.futureSavingsAndRetirementIncome.total[0]
  const afterContributions = {
    balance: start.balance + contributionsTotal,
  }
  const withdrawals = (() => {
    const regular = byMonthsFromNowByRun.withdrawals.regular[0][0]
    const essential = byMonthsFromNowByRun.withdrawals.essential[0][0]
    const discretionary = byMonthsFromNowByRun.withdrawals.discretionary[0][0]
    const total = regular + essential + discretionary

    const fromContributions = Math.min(total, contributionsTotal)
    const fromSavingsPortfolio = total - fromContributions
    return {
      regular,
      essential,
      discretionary,
      total,
      fromContributions,
      fromSavingsPortfolio,
    }
  })()

  const contributions = {
    total: contributionsTotal,
    toSavingsPortfolio: contributionsTotal - withdrawals.fromContributions,
    toWithdrawal: withdrawals.fromContributions,
  }

  const afterWithdrawals = {
    allocation: {
      stocks: _.round(
        byMonthsFromNowByRun.afterWithdrawals.allocation.stocks[0][0],
        4,
      ),
    },
    balance: afterContributions.balance - withdrawals.total,
  }
  const contributionToOrWithdrawalFromSavingsPortfolio =
    contributions.toSavingsPortfolio > 0
      ? {
          type: 'contribution' as const,
          contribution: contributions.toSavingsPortfolio,
        }
      : {
          type: 'withdrawal' as const,
          withdrawal: withdrawals.fromSavingsPortfolio,
        }

  return {
    start,
    contributions,
    afterContributions,
    withdrawals,
    afterWithdrawals,
    contributionToOrWithdrawalFromSavingsPortfolio,
  }
}
