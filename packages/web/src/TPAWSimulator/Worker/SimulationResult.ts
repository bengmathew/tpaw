import _ from 'lodash'
import {fGet} from '../../Utils/Utils'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {SavingsPortfolioThroughAYear} from './SavingsPortfolioThroughAYear'

export type SimulationResult<
  T extends {savingsPortfolio: SavingsPortfolioThroughAYear.End} = {
    savingsPortfolio: SavingsPortfolioThroughAYear.End
  }
> = {
  byYearFromNow: T[]
  endingBalanceOfSavingsPortfolio: number
  legacy: number
}

export const simulationResult = <
  T extends {savingsPortfolio: SavingsPortfolioThroughAYear.End}
>(
  byYearFromNow: T[],
  params: TPAWParamsProcessed
): SimulationResult<T> => {
  const lastYear = fGet(_.last(byYearFromNow))
  const endingBalanceOfSavingsPortfolio = lastYear.savingsPortfolio.end.balance
  const legacy = endingBalanceOfSavingsPortfolio + params.legacy.external
  return {byYearFromNow, endingBalanceOfSavingsPortfolio, legacy}
}
