import {
  PlanParams,
  noCase,
  partialDefaultDatelessPlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'

type ExpectedReturnsForPlanning =
  PlanParams['advanced']['returnsStatsForPlanning']['expectedValue']['empiricalAnnualNonLog']

type _CustomType = Extract<ExpectedReturnsForPlanning, { type: 'custom' }>

export const getExpectedReturnCustomStockBaseLabel = (
  base: _CustomType['stocks']['base'],
) => {
  switch (base) {
    case 'regressionPrediction':
      return {
        titleCase: 'Regression Prediction',
        lowercase: 'regression prediction',
      }
    case 'conservativeEstimate':
      return {
        titleCase: 'Conservative Estimate',
        lowercase: 'conservative estimate',
      }
    case '1/CAPE':
      return {
        titleCase: '1/CAPE',
        lowercase: '1/CAPE',
      }
    case 'historical':
      return {
        titleCase: 'Historical',
        lowercase: 'historical',
      }
    default:
      noCase(base)
  }
}
export const getExpectedReturnCustomBondBaseLabel = (
  base: _CustomType['bonds']['base'],
) => {
  switch (base) {
    case '20YearTIPSYield':
      return {
        titleCase: '20 Year TIPS Yield',
        lowercase: '20 year TIPS yield',
      }
    case 'historical':
      return {
        titleCase: 'Historical',
        lowercase: 'historical',
      }
    default:
      noCase(base)
  }
}

export const getExpectedReturnTypeLabelInfo = ({
  type,
}: {
  type: ExpectedReturnsForPlanning['type']
}):
  | { isSplit: true; stocks: string; bonds: string; forUndoRedo: string }
  | { isSplit: false; stocksAndBonds: string; forUndoRedo: string } => {
  const tipsYield20Year = '20 Year TIPS Yield'
  const withTIPSBonds = (stocks: string, stocksForUndoRedo: string) => ({
    isSplit: true as const,
    stocks,
    bonds: tipsYield20Year,
    forUndoRedo: `${stocksForUndoRedo} for stocks and 20 year TIPS yield for bonds`,
  })
  switch (type) {
    case 'conservativeEstimate,20YearTIPSYield':
      return withTIPSBonds('Conservative Estimate', 'conservative estimate')
    case '1/CAPE,20YearTIPSYield':
      return withTIPSBonds('1/CAPE', '1/CAPE')
    case 'regressionPrediction,20YearTIPSYield':
      return withTIPSBonds('Regression Prediction', 'regression prediction')
    case 'historical':
      return {
        isSplit: false,
        stocksAndBonds: 'Historical',
        forUndoRedo: 'historical',
      }
    case 'fixedEquityPremium':
      return {
        isSplit: false,
        stocksAndBonds: 'Fixed Equity Premium',
        forUndoRedo: 'fixed equity premium',
      }
    case 'custom':
      return {
        isSplit: false,
        stocksAndBonds: 'Custom',
        forUndoRedo: 'custom',
      }
    case 'fixed':
      return { isSplit: false, stocksAndBonds: 'Fixed', forUndoRedo: 'fixed' }
    default:
      noCase(type)
  }
}

export const useIsPlanInputExpectedReturnsAndVolatilityModified = () => {
  const isExpectedCardModified = useIsPlanInputExpectedReturnsCardModified()
  const isBondVolatilityModified = useIsPlanInputBondVolatilityCardModified()
  const isStockVolatilityModified = useIsPlanInputStockVolatilityCardModified()

  return (
    isExpectedCardModified ||
    isBondVolatilityModified ||
    isStockVolatilityModified
  )
}

export const useIsPlanInputExpectedReturnsCardModified = () => {
  const { planParamsNorm } = useSimulation()
  return !_.isEqual(
    partialDefaultDatelessPlanParams.advanced.returnsStatsForPlanning
      .expectedValue,
    planParamsNorm.advanced.returnsStatsForPlanning.expectedValue,
  )
}
export const useIsPlanInputStockVolatilityCardModified = () => {
  const { planParamsNorm } = useSimulation()
  return !_.isEqual(
    partialDefaultDatelessPlanParams.advanced.returnsStatsForPlanning
      .standardDeviation.stocks,
    planParamsNorm.advanced.returnsStatsForPlanning.standardDeviation.stocks,
  )
}

export const useIsPlanInputBondVolatilityCardModified = () => {
  const { planParamsNorm } = useSimulation()
  return !_.isEqual(
    partialDefaultDatelessPlanParams.advanced.historicalReturnsAdjustment
      .standardDeviation.bonds,
    planParamsNorm.advanced.historicalReturnsAdjustment.standardDeviation.bonds,
  )
}
