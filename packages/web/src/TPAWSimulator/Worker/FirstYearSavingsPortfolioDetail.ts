import {PlanParamsProcessed} from '../PlanParamsProcessed'
import {TPAWWorkerRunSimulationResult} from './TPAWWorkerTypes'

export type FirstYearSavingsPortfolioDetail = {
  start: {balance: number}
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
    allocation: {stocks: number}
    balance: number
  }
  contributionToOrWithdrawalFromSavingsPortfolio:
    | {type: 'contribution'; contribution: number}
    | {type: 'withdrawal'; withdrawal: number}
}

export const firstYearSavingsPortfolioDetail = (
  byYearsFromNowByRun: TPAWWorkerRunSimulationResult['byYearsFromNowByRun']['savingsPortfolio'],
  params: PlanParamsProcessed
): FirstYearSavingsPortfolioDetail => {
  const start = {
    balance: byYearsFromNowByRun.start.balance[0][0],
  }
  const contributionsTotal = params.byYear[0].futureSavingsAndRetirementIncome
  const afterContributions = {
    balance: start.balance + contributionsTotal,
  }
  const withdrawals = (() => {
    const regular = byYearsFromNowByRun.withdrawals.regular[0][0]
    const essential = byYearsFromNowByRun.withdrawals.essential[0][0]
    const discretionary = byYearsFromNowByRun.withdrawals.discretionary[0][0]
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
      stocks: byYearsFromNowByRun.afterWithdrawals.allocation.stocks[0][0],
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
