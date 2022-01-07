import {
  numericYear,
  TPAWParams,
  ValueForYearRange,
} from '../../../../TPAWSimulator/TPAWParams'
import {noCase} from '../../../../Utils/Utils'

type _Type = 'futureSavings' | 'retirementIncome' | 'extraSpending'
export const paramsInputValidate = (params: TPAWParams, type: _Type) => {
  const entries = (() => {
    switch (type) {
      case 'futureSavings':
        return params.savings
      case 'retirementIncome':
        return params.retirementIncome
      case 'extraSpending':
        return [
          ...params.withdrawals.fundedByBonds,
          ...params.withdrawals.fundedByRiskPortfolio,
        ]
    }
  })()

  return entries.every(
    x => paramsInputValidateYearRange(params, type, x.yearRange) === 'ok'
  )
}

export const paramsInputValidateYearRange = (
  {age}: TPAWParams,
  type: _Type,
  yearRange: ValueForYearRange['yearRange']
) => {
  const start = numericYear({age}, yearRange.start)
  const end = numericYear({age}, yearRange.end)
  const [min, max] = (() => {
    switch (type) {
      case 'futureSavings':
        return [age.start, age.retirement - 1]
      case 'retirementIncome':
        return [age.retirement, age.end]
      case 'extraSpending':
        return [age.start, age.end]
      default:
        noCase(type)
    }
  })()
  return start < min || end > max
    ? ('outOfBounds' as const)
    : start > end
    ? ('startGreaterThanEnd' as const)
    : ('ok' as const)
}
