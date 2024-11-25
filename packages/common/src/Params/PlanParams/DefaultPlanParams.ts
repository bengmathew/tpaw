import _ from 'lodash'
import { CalendarMonthFns } from '../../Misc/CalendarMonthFns'
import {
  CalendarDay,
  CalendarMonth,
  currentPlanParamsVersion,
  PlanParams,
} from './PlanParams'
import { getLastMarketDataDayForUndatedPlans } from './GetLastMarketDataDayForUndatedPlans'

// TODO: Remove in favor of Rust version
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

export const getFullDatedDefaultPlanParams = (
  timestamp: number,
  ianaTimezoneName: string,
): PlanParams => ({
  ...getPartialDefaultPlanParams({
    isDatedPlan: true,
    nowAsCalendarMonth: CalendarMonthFns.fromTimestamp(
      timestamp,
      ianaTimezoneName,
    ),
  }),
  timestamp,
  datingInfo: { isDated: true },
})

export const getFullDatelessDefaultPlanParams = (
  timestamp: number,
): PlanParams => ({
  ...getPartialDefaultPlanParams({ isDatedPlan: false }),
  timestamp,
  datingInfo: {
    isDated: false,
    marketDataAsOfEndOfDayInNY: getLastMarketDataDayForUndatedPlans(timestamp),
  },
})

export const getPartialDefaultPlanParams = (
  props:
    | {
        isDatedPlan: true
        nowAsCalendarMonth: CalendarMonth
      }
    | { isDatedPlan: false },
): Omit<PlanParams, 'timestamp' | 'datingInfo'> => ({
  v: currentPlanParamsVersion,
  dialogPositionNominal: 'age',
  people: {
    withPartner: false,
    person1: {
      ages: {
        type: 'retirementDateSpecified',
        currentAgeInfo: props.isDatedPlan
          ? {
              isDatedPlan: props.isDatedPlan,
              monthOfBirth: CalendarMonthFns.addMonths(
                props.nowAsCalendarMonth,
                -35 * 12,
              ),
            }
          : {
              isDatedPlan: props.isDatedPlan,
              currentAge: { inMonths: 35 * 12 },
            },
        retirementAge: { inMonths: 65 * 12 },
        maxAge: { inMonths: 100 * 12 },
      },
    },
  },

  wealth: {
    portfolioBalance: props.isDatedPlan
      ? {
          isDatedPlan: props.isDatedPlan,
          updatedHere: true,
          amount: 0,
        }
      : {
          isDatedPlan: props.isDatedPlan,
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
          month: {
            type: 'now',
            monthOfEntry: props.isDatedPlan
              ? {
                  isDatedPlan: props.isDatedPlan,
                  calendarMonth: props.nowAsCalendarMonth,
                }
              : { isDatedPlan: props.isDatedPlan },
          },
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
    returnsStatsForPlanning: {
      expectedValue: {
        empiricalAnnualNonLog: {
          type: 'regressionPrediction,20YearTIPSYield',
        },
      },
      standardDeviation: {
        stocks: { scale: { log: 1 } },
        bonds: { scale: { log: 0 } },
      },
    },
    historicalReturnsAdjustment: {
      standardDeviation: {
        bonds: { scale: { log: 1 } },
      },
      overrideToFixedForTesting: { type: 'none' },
    },
    annualInflation: { type: 'suggested' },
    sampling: {
      type: 'monteCarlo',
      data: {
        blockSize: { inMonths: 12 * 5 },
        staggerRunStarts: true,
      },
    },
    strategy: 'TPAW',
  },
  results: null,
})

export const partialDefaultDatelessPlanParams = getPartialDefaultPlanParams({
  isDatedPlan: false,
})
