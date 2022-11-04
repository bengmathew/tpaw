import _ from 'lodash'
import { historicalReturnsAverage } from '../HistoricalReturns'
import { linearFnFomPoints, noCase } from '../Utils'
import { PlanParams, TPAWRisk, TPAWRiskLevel } from './PlanParams'

type MarketData = {
  CAPE: {
    suggested: number
    oneOverCAPE: number
    regressionAverage: number
  }
  bondRates: { twentyYear: number }
  inflation: { value: number }
}
1
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
      return { ...historicalReturnsAverage }
    default:
      noCase(type)
  }
}

export const SUGGESTED_INFLATION = (marketData: MarketData) =>
  _.round(marketData.inflation.value, 3)

export const TPAW_RISK_PRESETS = (
  preset: Exclude<TPAWRiskLevel, 'custom'>,
  numYears: number,
): TPAWRisk => {
  const allocFn = (at20of100: number, spendingTilt: number): TPAWRisk => {
    const at100of100 = _.round(at20of100 - 0.1, 2)
    return {
      tpaw: {
        allocation: {
          start: {
            stocks: _.round(
              linearFnFomPoints(0, at100of100, 100 - 20, at20of100)(numYears),
              2,
            ),
          },
          intermediate: [],
          end: { stocks: at100of100 },
        },
        allocationForLegacy: { stocks: _.round(at20of100 + 0.2, 2) },
      },
      tpawAndSPAW: {
        spendingCeiling: null,
        spendingFloor: null,
        spendingTilt,
        lmp: 0,
      },
    }
  }
  switch (preset) {
    case 'riskLevel-1':
      return allocFn(0.3, 0.005)
    case 'riskLevel-2':
      return allocFn(0.4, 0.008)
    case 'riskLevel-3':
      return allocFn(0.5, 0.011)
    case 'riskLevel-4':
      return allocFn(0.6, 0.014)
    default:
      noCase(preset)
  }
}

export const resolveTPAWRiskPreset = (
  risk: PlanParams['risk'],
  numYears: number,
): Exclude<PlanParams['risk'], { useTPAWPreset: true }> => {
  if (!risk.useTPAWPreset) return risk
  const { swr, spawAndSWR, tpawPreset, customTPAWPreset, savedTPAWPreset } =
    risk
  const { tpaw, tpawAndSPAW } = TPAW_RISK_PRESETS(risk.tpawPreset, numYears)
  return {
    useTPAWPreset: false,
    tpaw,
    tpawAndSPAW,
    swr,
    spawAndSWR,
    tpawPreset,
    customTPAWPreset,
    savedTPAWPreset,
  }
}

export const DEFAULT_SWR_WITHDRAWAL_PERCENT = (retirementLength: number) => {
  return _.round(0.7125 * Math.pow(retirementLength, -0.859), 3)
}

export function getDefaultPlanParams() {
  const params: PlanParams = {
    v: 14,
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
      savedTPAWPreset: null,
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
