import {blendReturns} from '../../Utils/BlendReturns'
import {assert} from '../../Utils/Utils'

export namespace SavingsPortfolioThroughAYear {
  export type Start = {start: {balance: number}}
  export type AfterContributions = ReturnType<typeof applyContributions>
  export type AfterWithdrawals = ReturnType<typeof applyWithdrawals>
  export type End = ReturnType<typeof applyAllocation>

  export const applyContributions = (contributions: number, {start}: Start) => {
    const balance = start.balance + contributions

    return {
      start,
      contributions: {total: contributions},
      afterContributions: {
        balance,
      },
    }
  }

  export const applyWithdrawals = (
    withdrawals: {
      essential: number
      discretionary: number
      regular: number
    },
    {start, contributions, afterContributions}: AfterContributions
  ) => {
    const totalWithdrawal =
      withdrawals.essential + withdrawals.discretionary + withdrawals.regular
    assert(totalWithdrawal <= afterContributions.balance)
    const balance = afterContributions.balance - totalWithdrawal

    const fromContributions = Math.min(totalWithdrawal, contributions.total)
    const fromSavingsPortfolio = totalWithdrawal - fromContributions
    const fromSavingsPortfolioRate =
      start.balance === 0 ? 0 : fromSavingsPortfolio / start.balance

    const contributionToSavingsPortfolio =
      contributions.total - fromContributions
    return {
      start,
      contributions: {
        ...contributions,
        toSavingsPortfolio: contributionToSavingsPortfolio,
        toWithdrawal: fromContributions,
      },
      afterContributions,
      withdrawals: {
        ...withdrawals,
        total: totalWithdrawal,
        fromContributions,
        fromSavingsPortfolio,
        fromSavingsPortfolioRate,
      },
      contributionToOrWithdrawalFromSavingsPortfolio:
        contributionToSavingsPortfolio > 0
          ? ({
              type: 'contribution',
              contribution: contributionToSavingsPortfolio,
            } as const)
          : ({
              type: 'withdrawal',
              withdrawal: fromSavingsPortfolio,
            } as const),
      afterWithdrawals: {
        balance,
      },
    }
  }

  export const applyAllocation = (
    allocation: {stocks: number},
    returnRate: {stocks: number; bonds: number},
    state: AfterWithdrawals
  ) => {
    const balance =
      state.afterWithdrawals.balance *
      (1 + blendReturns(returnRate)(allocation))
    return {
      ...state,
      afterWithdrawals: {...state.afterWithdrawals, allocation, returnRate},
      end: {balance},
    }
  }
}
