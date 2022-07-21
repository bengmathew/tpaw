import _ from 'lodash'
import {noCase} from '../Utils/Utils'
import {historicalReturnsAverage} from './HistoricalReturns'
import {TPAWParams} from './TPAWParams'

export const EXPECTED_RETURN_PRESETS = (
  type: Exclude<TPAWParams['returns']['expected']['type'], 'manual'>
) => {
  const suggested = {stocks: 0.043, bonds: 0.007}
  switch (type) {
    case 'suggested':
      return {...suggested}
    case 'oneOverCAPE':
      return {
        stocks: 0.035,
        bonds: suggested.bonds,
      }
    case 'regressionPrediction':
      return {
        stocks: 0.055,
        bonds: suggested.bonds,
      }
    case 'historical':
      return {...historicalReturnsAverage}
    default:
      noCase(type)
  }
}

export const SUGGESTED_INFLATION = 0.024

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
          start: {stocks: 0.40},
          intermediate: [],
          end: {stocks: 0.30},
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
