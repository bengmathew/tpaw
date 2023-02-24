import _ from 'lodash'
import { historicalReturns } from '../HistoricalReturns/HistoricalReturns'
import { noCase } from '../Utils'
import { PlanParams } from './PlanParams'

type MarketData = {
  CAPE: {
    suggested: number
    oneOverCAPE: number
    regressionAverage: number
  }
  bondRates: { twentyYear: number }
  inflation: { value: number }
}

export const EXPECTED_ANNUAL_RETURN_PRESETS = (
  type: Exclude<
    PlanParams['advanced']['annualReturns']['expected']['type'],
    'manual'
  >,
  { CAPE, bondRates }: MarketData,
) => {
  const suggested = {
    stocks: _.round(CAPE.suggested, 3),
    bonds: _.round(bondRates.twentyYear, 3),
  }
  switch (type) {
    case 'suggested':
      return { ...suggested }
    case 'oneOverCAPE':
      return {
        stocks: _.round(CAPE.oneOverCAPE, 3),
        bonds: suggested.bonds,
      }
    case 'regressionPrediction':
      return {
        stocks: _.round(CAPE.regressionAverage, 3),
        bonds: suggested.bonds,
      }
    case 'historical':
      // Intentionally not rounding here.
      return {
        stocks: historicalReturns.monthly.annualStats.stocks.mean,
        bonds: historicalReturns.monthly.annualStats.bonds.mean,
      }
    default:
      noCase(type)
  }
}

export const SUGGESTED_ANNUAL_INFLATION = (marketData: MarketData) =>
  _.round(marketData.inflation.value, 3)

export const DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT = (
  retirementLengthInMonths: number,
) => {
  return _.round(0.7125 * Math.pow(retirementLengthInMonths / 12, -0.859), 3)
}

export function getDefaultPlanParams() {
  const params: PlanParams = {
    v: 19,
    warnedAbout14to15Converstion: true,
    warnedAbout16to17Converstion: true,
    dialogPosition: 'age',
    people: {
      withPartner: false,
      person1: {
        ages: {
          type: 'notRetired',
          currentMonth: 35 * 12,
          retirementMonth: 65 * 12,
          maxMonth: 100 * 12,
        },
      },
    },

    wealth: {
      currentPortfolioBalance: 0,
      futureSavings: [],
      retirementIncome: [],
    },

    adjustmentsToSpending: {
      tpawAndSPAW: {
        monthlySpendingCeiling: null,
        monthlySpendingFloor: null,
        legacy: {
          total: 0,
          external: [],
        },
      },
      extraSpending: {
        essential: [],
        discretionary: [],
      },
    },

    risk: {
      tpaw: {
        riskTolerance: {
          at20: 12,
          deltaAtMaxAge: -2,
          forLegacyAsDeltaFromAt20: 2,
        },
        timePreference: 0,
        additionalAnnualSpendingTilt: 0,
      },
      tpawAndSPAW: {
        lmp: 0,
      },
      spaw: {
        annualSpendingTilt: 0.008,
      },

      spawAndSWR: {
        allocation: {
          start: { stocks: 0.5 },
          intermediate: [],
          end: { stocks: 0.5 },
        },
      },
      swr: {
        withdrawal: { type: 'default' },
      },
    },

    advanced: {
      annualReturns: {
        expected: { type: 'suggested' },
        historical: {
          type: 'adjusted',
          adjustment: { type: 'toExpected' },
          correctForBlockSampling: true,
        },
      },
      annualInflation: { type: 'suggested' },
      sampling: 'monteCarlo',
      samplingBlockSizeForMonteCarlo: 12 * 5,
      strategy: 'TPAW',
    },
    dev: {
      alwaysShowAllMonths: false,
    },
  }

  return params
}
