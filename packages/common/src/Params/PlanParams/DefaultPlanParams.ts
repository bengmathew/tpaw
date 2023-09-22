import _ from 'lodash'
import { DateTime } from 'luxon'
import { historicalReturns } from '../../HistoricalReturns/HistoricalReturns'
import { noCase } from '../../Utils'
import { calendarMonthFromTime, PlanParams } from './PlanParams'

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

export function getDefaultPlanParams(
  currentTimestamp: number,
  ianaTimezoneName: string,
): PlanParams {
  // const { getMonthOfBirth, currentMonth } =
  //   getCurrentTimeFnsExt(currentTimeInfo)

  const currentDateTime =
    DateTime.fromMillis(currentTimestamp).setZone(ianaTimezoneName)

  const params: PlanParams = {
    v: 22,
    timestamp: currentTimestamp,
    dialogPosition: 'age',
    people: {
      withPartner: false,
      person1: {
        ages: {
          type: 'retirementDateSpecified',
          monthOfBirth: calendarMonthFromTime(
            currentDateTime.minus({ month: 35 * 12 }),
          ),
          retirementAge: { inMonths: 65 * 12 },
          maxAge: { inMonths: 100 * 12 },
        },
      },
    },

    wealth: {
      portfolioBalance: {
        updatedHere: true,
        amount: 0,
      },
      futureSavings: {},
      incomeDuringRetirement: {},
    },

    adjustmentsToSpending: {
      tpawAndSPAW: {
        monthlySpendingCeiling: null,
        monthlySpendingFloor: null,
        legacy: {
          total: 0,
          external: {},
        },
      },
      extraSpending: {
        essential: {},
        discretionary: {},
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
          start: {
            month: calendarMonthFromTime(currentDateTime),
            stocks: 0.5,
          },
          intermediate: {},
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
          stocks: {
            type: 'adjustExpected',
            adjustment: { type: 'toExpectedUsedForPlanning' },
            correctForBlockSampling: true,
          },
          bonds: {
            type: 'adjustExpected',
            adjustment: { type: 'toExpectedUsedForPlanning' },
            correctForBlockSampling: true,
          },
        },
      },
      annualInflation: { type: 'suggested' },
      sampling: { type: 'monteCarlo', blockSizeForMonteCarloSampling: 12 * 5 },
      strategy: 'TPAW',
    },
    results: null,
  }

  return params
}
