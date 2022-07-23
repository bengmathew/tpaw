import _ from 'lodash'
import {MarketData} from '../Pages/Common/GetMarketData'
import {noCase} from '../Utils/Utils'
import {historicalReturnsAverage} from './HistoricalReturns'
import {TPAWParams} from './TPAWParams'

export const EXPECTED_RETURN_PRESETS = (
  type: Exclude<TPAWParams['returns']['expected']['type'], 'manual'>,
  {CAPE, bondRates}: MarketData
) => {
  const suggested = {
    stocks: _.round(CAPE.suggested, 3),
    bonds: _.round(bondRates.twentyYear, 3),
  }
  switch (type) {
    case 'suggested':
      return {...suggested}
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
      return {...historicalReturnsAverage}
    default:
      noCase(type)
  }
}

export const SUGGESTED_INFLATION = (marketData: MarketData) =>
  _.round(marketData.inflation.value, 3)

export function getDefaultParams() {
  const params: TPAWParams = {
    v: 12,
    strategy: 'TPAW',
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 35, retirement: 65, max: 100},
      },
    },
    returns: {
      expected: {type: 'suggested'},
      historical: {type: 'default', adjust: {type: 'toExpected'}},
    },
    inflation: {type: 'suggested'},
    targetAllocation: {
      regularPortfolio: {
        forTPAW: {
          start: {stocks: 0.4},
          intermediate: [],
          end: {stocks: 0.3},
        },
        forSPAWAndSWR: {
          start: {stocks: 0.5},
          intermediate: [],
          end: {stocks: 0.5},
        },
      },
      legacyPortfolio: {stocks: 0.7},
    },
    swrWithdrawal: {type: 'default'},
    scheduledWithdrawalGrowthRate: 0.01,
    savingsAtStartOfStartYear: 100000,
    savings: [],
    retirementIncome: [],
    withdrawals: {
      lmp: 0,
      essential: [],
      discretionary: [],
    },
    spendingCeiling: null,
    spendingFloor: null,
    legacy: {
      total: 0,
      external: [],
    },
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  }

  return params
}

export const defaultSWRWithdrawalRate = (retirementLength: number) => {
  return _.round(0.7125 * Math.pow(retirementLength, -0.859), 3)
}
