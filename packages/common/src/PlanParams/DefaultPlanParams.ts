import _ from 'lodash'
import { historicalReturns } from '../HistoricalReturns'
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

export const EXPECTED_RETURN_PRESETS = (
  type: Exclude<PlanParams['returns']['expected']['type'], 'manual'>,
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
      return {
        stocks: historicalReturns.stocks.expectedValue,
        bonds: historicalReturns.bonds.expectedValue,
      }
    default:
      noCase(type)
  }
}

export const SUGGESTED_INFLATION = (marketData: MarketData) =>
  _.round(marketData.inflation.value, 3)

export const DEFAULT_SWR_WITHDRAWAL_PERCENT = (retirementLength: number) => {
  return _.round(0.7125 * Math.pow(retirementLength, -0.859), 3)
}

export function getDefaultPlanParams() {
  const params: PlanParams = {
    v: 15,
    warnedAbout14to15Converstion: true,
    strategy: 'TPAW',
    dialogMode: true,
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: { type: 'notRetired', current: 35, retirement: 65, max: 100 },
      },
    },
    currentPortfolioBalance: 0,
    futureSavings: [],
    retirementIncome: [],

    adjustmentsToSpending: {
      tpawAndSPAW: {
        spendingCeiling: null,
        spendingFloor: null,
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
      },
      tpawAndSPAW: {
        lmp: 0,
      },
      spaw: {
        spendingTilt: 0.008,
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

    returns: {
      expected: { type: 'suggested' },
      historical: { type: 'default', adjust: { type: 'toExpected' } },
    },
    inflation: { type: 'suggested' },
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  }

  return params
}
