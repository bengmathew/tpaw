import {
  TPAWParams,
  ValueForYearRange,
  Year,
} from '../../../../TPAWSimulator/TPAWParams'

export function mapAllValueForYearsRangesInParams<T>(
  params: TPAWParams,
  fn: (
    range: ValueForYearRange,
    type: 'future-savings' | 'income-during-retirement' | 'extra-spending',
    params: TPAWParams,
    yearUpdater?: (year: Year) => Year
  ) => T,
  yearUpdater?: (year: Year) => Year
) {
  const {savings, retirementIncome, withdrawals} = params
  return [
    ...savings.map(x => fn(x, 'future-savings', params, yearUpdater)),
    ...retirementIncome.map(x =>
      fn(x, 'income-during-retirement', params, yearUpdater)
    ),
    ...withdrawals.fundedByBonds.map(x =>
      fn(x, 'extra-spending', params, yearUpdater)
    ),
    ...withdrawals.fundedByRiskPortfolio.map(x =>
      fn(x, 'extra-spending', params, yearUpdater)
    ),
  ]
}
