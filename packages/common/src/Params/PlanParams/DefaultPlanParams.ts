import _ from 'lodash'
import { CalendarMonthFns } from '../../Misc/CalendarMonthFns'
import { currentPlanParamsVersion, PlanParams } from './PlanParams'

export const DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT = (
  retirementLengthInMonths: number,
) => {
  return _.round(0.7125 * Math.pow(retirementLengthInMonths / 12, -0.859), 3)
}

export const DEFAULT_RISK_TPAW = {
  riskTolerance: {
    at20: 12,
    deltaAtMaxAge: -2,
    forLegacyAsDeltaFromAt20: 2,
  },
  timePreference: 0,
  additionalAnnualSpendingTilt: 0,
}

export function getDefaultPlanParams(
  currentTimestamp: number,
  ianaTimezoneName: string,
): PlanParams {
  const nowAsCalendarMonth = CalendarMonthFns.fromTimestamp(
    currentTimestamp,
    ianaTimezoneName,
  )

  const params: PlanParams = {
    v: currentPlanParamsVersion,
    timestamp: currentTimestamp,
    dialogPositionNominal: 'age',
    people: {
      withPartner: false,
      person1: {
        ages: {
          type: 'retirementDateSpecified',
          monthOfBirth: CalendarMonthFns.addMonths(
            nowAsCalendarMonth,
            -35 * 12,
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
      tpaw: DEFAULT_RISK_TPAW,
      tpawAndSPAW: {
        lmp: 0,
      },
      spaw: {
        annualSpendingTilt: 0.008,
      },

      spawAndSWR: {
        allocation: {
          start: {
            month: nowAsCalendarMonth,
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
      expectedReturnsForPlanning: {
        type: 'regressionPrediction,20YearTIPSYield',
      },
      historicalMonthlyLogReturnsAdjustment: {
        standardDeviation: {
          stocks: { scale: 1 },
          bonds: { enableVolatility: true },
        },
        overrideToFixedForTesting: false,
      },
      annualInflation: { type: 'suggested' },
      sampling: {
        type: 'monteCarlo',
        forMonteCarlo: {
          blockSize: { inMonths: 12 * 5 },
          staggerRunStarts: true,
        },
      },
      strategy: 'TPAW',
    },
    results: null,
  }

  return params
}
