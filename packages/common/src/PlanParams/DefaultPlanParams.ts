import _ from 'lodash'
import { DateTime } from 'luxon'
import { historicalReturns } from '../HistoricalReturns/HistoricalReturns'
import { noCase } from '../Utils'
import { calendarMonthFromTime, Params, PlanParams } from './Params'

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

export function getDefaultPlanParams(currentTime: DateTime) {
  // const { getMonthOfBirth, currentMonth } =
  //   getCurrentTimeFnsExt(currentTimeInfo)

  const params: Params = {
    v: 20,
    plan: {
      timestamp: currentTime.valueOf(),
      dialogPosition: 'age',
      people: {
        withPartner: false,
        person1: {
          ages: {
            type: 'retirementDateSpecified',
            monthOfBirth: calendarMonthFromTime(
              currentTime.minus({ month: 35 * 12 }),
            ),
            retirementAge: { inMonths: 65 * 12 },
            maxAge: { inMonths: 100 * 12 },
          },
        },
      },

      wealth: {
        portfolioBalance: {
          isLastPlanChange: true,
          amount: 0,
          timestamp: currentTime.valueOf(),
        },
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
            start: { month: calendarMonthFromTime(currentTime), stocks: 0.5 },
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
        monteCarloSampling: {
          blockSize: 12 * 5,
          numOfSimulations: 500,
        },

        strategy: 'TPAW',
      },
    },
    nonPlan: {
      migrationWarnings: {
        v14tov15: true,
        v16tov17: true,
        v19tov20: true,
      },
      percentileRange: { start: 5, end: 95 },
      defaultTimezone: {
        type: 'auto',
        ianaTimezoneName: currentTime.zoneName,
      },
      dev: {
        alwaysShowAllMonths: false,
        currentTimeFastForward: { shouldFastForward: false },
      },
    },
  }

  return params
}
