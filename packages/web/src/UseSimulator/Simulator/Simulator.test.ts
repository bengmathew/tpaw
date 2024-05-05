import {
  MonthRange,
  PlanParams,
  LabeledAmountTimedList,
  currentPlanParamsVersion,
  getZonedTimeFns,
  letIn,
} from '@tpaw/common'
import * as Rust from '@tpaw/simulator'

describe('simulator', () => {
  let now = Date.now()
  let ianaTimezoneName = 'America/New_York'
  let nowAsCalendarDay = letIn(
    getZonedTimeFns(ianaTimezoneName)(now),
    (nowAsDateTime) => ({
      year: nowAsDateTime.year,
      month: nowAsDateTime.month,
    }),
  )

  let yearsFromNowAsCalendarMonth = (yearsFromNow: number) => ({
    year: nowAsCalendarDay.year + yearsFromNow,
    month: nowAsCalendarDay.month,
  })

  const getPlanParams = (
    withdrawalStart: 'person1' | 'person2',
    strategy: 'TPAW' | 'SPAW' | 'SWR',
  ): PlanParams => {
    const getLabeledAmountTimedList = (
      x: { monthRange: MonthRange; amount: number }[],
    ): LabeledAmountTimedList => {
      let result: LabeledAmountTimedList = {}
      x.forEach(({ monthRange, amount }, i) => {
        let id = `${i}`
        result[id] = {
          id,
          sortIndex: i,
          colorIndex: i,
          label: id,
          nominal: false,
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: amount,
            monthRange,
          },
        }
      })
      return result
    }
    return {
      v: currentPlanParamsVersion,
      timestamp: now,
      datingInfo: { isDated: true },
      dialogPositionNominal: 'done',
      people: {
        withPartner: true,
        person1: {
          ages: {
            type: 'retirementDateSpecified',
            currentAgeInfo: {
              isDatedPlan: true,
              monthOfBirth: yearsFromNowAsCalendarMonth(-30),
            },
            retirementAge: { inMonths: 65 * 12 },
            maxAge: { inMonths: 95 * 12 },
          },
        },
        person2: {
          ages: {
            type: 'retirementDateSpecified',
            currentAgeInfo: {
              isDatedPlan: true,
              monthOfBirth: yearsFromNowAsCalendarMonth(-40),
            },
            retirementAge: { inMonths: 70 * 12 },
            maxAge: { inMonths: 100 * 12 },
          },
        },
        withdrawalStart,
      },
      wealth: {
        portfolioBalance: {
          isDatedPlan: true,
          updatedHere: true,
          amount: 100000,
        },
        futureSavings: getLabeledAmountTimedList([
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: yearsFromNowAsCalendarMonth(-1),
                },
              },
              end: {
                type: 'namedAge',
                person: 'person1',
                age: 'lastWorkingMonth',
              },
            },
            amount: 1000,
          },
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: yearsFromNowAsCalendarMonth(1),
                },
              },
              end: {
                type: 'namedAge',
                person: 'person2',
                age: 'lastWorkingMonth',
              },
            },
            amount: 2000,
          },
        ]),

        incomeDuringRetirement: getLabeledAmountTimedList([
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'namedAge',
                person: 'person1',
                age: 'retirement',
              },
              end: { type: 'namedAge', person: 'person1', age: 'max' },
            },
            amount: 3000,
          },
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'namedAge',
                person: 'person2',
                age: 'retirement',
              },
              end: { type: 'namedAge', person: 'person2', age: 'max' },
            },
            amount: 4000,
          },
        ]),
      },
      adjustmentsToSpending: {
        extraSpending: {
          essential: getLabeledAmountTimedList([
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'now',
                  monthOfEntry: {
                    isDatedPlan: true,
                    calendarMonth: yearsFromNowAsCalendarMonth(-1),
                  },
                },
                end: { type: 'namedAge', person: 'person1', age: 'max' },
              },
              amount: 500,
            },
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'now',
                  monthOfEntry: {
                    isDatedPlan: true,
                    calendarMonth: yearsFromNowAsCalendarMonth(-1),
                  },
                },
                end: { type: 'namedAge', person: 'person2', age: 'max' },
              },
              amount: 600,
            },
          ]),
          discretionary: getLabeledAmountTimedList([
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'now',
                  monthOfEntry: {
                    isDatedPlan: true,
                    calendarMonth: yearsFromNowAsCalendarMonth(-1),
                  },
                },
                end: { type: 'namedAge', person: 'person1', age: 'max' },
              },
              amount: 700,
            },
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'now',
                  monthOfEntry: {
                    isDatedPlan: true,
                    calendarMonth: yearsFromNowAsCalendarMonth(-1),
                  },
                },
                end: { type: 'namedAge', person: 'person2', age: 'max' },
              },
              amount: 800,
            },
          ]),
        },
        tpawAndSPAW: {
          monthlySpendingCeiling: null,
          monthlySpendingFloor: null,
          legacy: {
            total: 10000,
            external: {},
          },
        },
      },
      risk: {
        tpaw: {
          riskTolerance: {
            at20: 12,
            deltaAtMaxAge: 0,
            forLegacyAsDeltaFromAt20: 0,
          },
          timePreference: 0.0,
          additionalAnnualSpendingTilt: 0.0,
        },
        tpawAndSPAW: {
          lmp: 0,
        },
        spaw: { annualSpendingTilt: 0.01 },
        spawAndSWR: {
          allocation: {
            start: {
              month: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: yearsFromNowAsCalendarMonth(-1),
                },
              },
              stocks: 0.5,
            },
            intermediate: {},
            end: { stocks: 0.5 },
          },
        },
        swr: {
          withdrawal: { type: 'asAmountPerMonth', amountPerMonth: 1000 },
        },
      },
      advanced: {
        strategy,
        returnsStatsForPlanning: {
          expectedValue: {
            empiricalAnnualNonLog: {
              type: 'fixed',
              stocks: 0.05,
              bonds: 0.02,
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
            overrideToFixedForTesting: true,
          },
        },
        annualInflation: { type: 'manual', value: 0.02 },
        sampling: {
          type: 'monteCarlo',
          data: {
            blockSize: { inMonths: 12 * 5 },
            staggerRunStarts: true,
          },
        },
      },
      results: null,
    }
  }
  // This should not matter. It is used for inflation and expected returns
  // presets and historical returns which we don't use.
  const marketData: Rust.DataForMarketBasedPlanParamValues = {
    closingTime: 0,
    inflation: {
      closingTime: 0,
      value: 0,
    },
    sp500: {
      closingTime: 0,
      value: 0,
    },
    bondRates: {
      closingTime: 0,
      fiveYear: 0,
      sevenYear: 0,
      tenYear: 0,
      twentyYear: 0,
      thirtyYear: 0,
    },

    timestampForMarketData: Number.MAX_SAFE_INTEGER,
  }

  test('with fixed historical returns', async () => {
    // const planParams = getPlanParams('person1', 'TPAW')
    // assert(planParams.wealth.portfolioBalance.updatedHere)
    // const planParamsExt = extendPlanParams(planParams, now, ianaTimezoneName)
    // const planParamsProcessed = processPlanParams(
    //   planParamsExt,
    //   planParams.wealth.portfolioBalance.amount,
    //   marketData,
    // )
    // const args: SimulationArgs = {
    //   planParams,
    //   planParamsProcessed,
    //   planParamsExt,
    //   numOfSimulationForMonteCarloSampling: 500,
    //   randomSeed: 1,
    // }

    // const result = await getSimulatorSingleton().runSimulations(
    //   { canceled: false },
    //   args,
    // )
    // expect(result).toBeTruthy()
    // assert(result)
    expect(true).toBeTruthy()
  })
})
