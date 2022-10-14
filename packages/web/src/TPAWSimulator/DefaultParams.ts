import _ from 'lodash'
import { MarketData } from '../Pages/Common/GetMarketData'
import { fGet, noCase } from '../Utils/Utils'
import { historicalReturnsAverage } from './HistoricalReturns'
import { TPAWParams, TPAWRisk, TPAWRiskLevel } from './TPAWParams'

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

export const TPAW_RISK_PRESETS = (
  preset: Exclude<TPAWRiskLevel, 'custom'>
): TPAWRisk => {
  switch (preset) {
    case 'riskLevel-1':
      return {
        tpaw: {
          allocation: {
            start: {stocks: 0.3},
            intermediate: [],
            end: {stocks: 0.2},
          },
          allocationForLegacy: {stocks: 0.5},
        },
        tpawAndSPAW: {
          spendingCeiling: null,
          spendingFloor: null,
          spendingTilt: 0.005,
          lmp: 0,
        },
      }
    case 'riskLevel-2':
      return {
        tpaw: {
          allocation: {
            start: {stocks: 0.4},
            intermediate: [],
            end: {stocks: 0.3},
          },
          allocationForLegacy: {stocks: 0.6},
        },
        tpawAndSPAW: {
          spendingCeiling: null,
          spendingFloor: null,
          spendingTilt: 0.008,
          lmp: 0,
        },
      }
    case 'riskLevel-3':
      return {
        tpaw: {
          allocation: {
            start: {stocks: 0.5},
            intermediate: [],
            end: {stocks: 0.4},
          },
          allocationForLegacy: {stocks: 0.7},
        },
        tpawAndSPAW: {
          spendingCeiling: null,
          spendingFloor: null,
          spendingTilt: 0.011,
          lmp: 0,
        },
      }
    case 'riskLevel-4':
      return {
        tpaw: {
          allocation: {
            start: {stocks: 0.6},
            intermediate: [],
            end: {stocks: 0.5},
          },
          allocationForLegacy: {stocks: 0.8},
        },
        tpawAndSPAW: {
          spendingCeiling: null,
          spendingFloor: null,
          spendingTilt: 0.014,
          lmp: 0,
        },
      }
    default:
      noCase(preset)
  }
}

export const resolveTPAWRiskPreset = (
  risk: TPAWParams['risk']
): Exclude<TPAWParams['risk'], {useTPAWPreset: true}> => {
  if (!risk.useTPAWPreset) return risk
  const {swr, spawAndSWR, tpawPreset, customTPAWPreset} = risk
  const {tpaw, tpawAndSPAW} =
    risk.tpawPreset === 'custom'
      ? fGet(customTPAWPreset)
      : TPAW_RISK_PRESETS(risk.tpawPreset)
  return {
    useTPAWPreset: false,
    tpaw,
    tpawAndSPAW,
    swr,
    spawAndSWR,
    tpawPreset,
    customTPAWPreset,
  }
}

export const DEFAULT_SWR_WITHDRAWAL_PERCENT = (retirementLength: number) => {
  return _.round(0.7125 * Math.pow(retirementLength, -0.859), 3)
}

export function getDefaultParams() {
  const params: TPAWParams = {
    v: 13,
    strategy: 'TPAW',
    dialogMode: true,
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 35, retirement: 65, max: 100},
      },
    },
    currentPortfolioBalance: 0,
    futureSavings: [],
    retirementIncome: [],
    extraSpending: {
      essential: [],
      discretionary: [],
    },
    legacy: {
      tpawAndSPAW: {
        total: 0,
        external: [],
      },
    },
    risk: {
      useTPAWPreset: true,
      tpawPreset: 'riskLevel-2',
      customTPAWPreset: null,
      spawAndSWR: {
        allocation: {
          start: {stocks: 0.5},
          intermediate: [],
          end: {stocks: 0.5},
        },
      },
      swr: {
        withdrawal: {type: 'default'},
      },
    },

    returns: {
      expected: {type: 'suggested'},
      historical: {type: 'default', adjust: {type: 'toExpected'}},
    },
    inflation: {type: 'suggested'},
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  }

  return params
}
