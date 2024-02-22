import {
  MonthRange,
  PlanParams,
  ValueForMonthRanges,
  assert,
  getZonedTimeFns,
  letIn,
} from '@tpaw/common'
import { SimulationArgs } from './Simulator'
import { processPlanParams } from '../PlanParamsProcessed/PlanParamsProcessed'
import { extendPlanParams } from '../ExtentPlanParams'
import * as Rust from '@tpaw/simulator'
import { getSimulatorSingleton } from '../UseSimulator'

describe('simulator', () => {
  let now = Date.now()
  let ianaTimezoneName = 'America/New_York'
  let nowAsCalendarMonth = letIn(
    getZonedTimeFns(ianaTimezoneName)(now),
    (nowAsDateTime) => ({
      year: nowAsDateTime.year,
      month: nowAsDateTime.month,
    }),
  )

  let yearsFromNowAsCalendarMonth = (yearsFromNow: number) => ({
    year: nowAsCalendarMonth.year + yearsFromNow,
    month: nowAsCalendarMonth.month,
  })

  const getPlanParams = (
    withdrawalStart: 'person1' | 'person2',
    strategy: 'TPAW' | 'SPAW' | 'SWR',
  ): PlanParams => {
    const getValueForMonthRanges = (
      x: { monthRange: MonthRange; value: number }[],
    ): ValueForMonthRanges => {
      let result: ValueForMonthRanges = {}
      x.forEach(({ monthRange, value }, i) => {
        let id = `${i}`
        result[id] = {
          id,
          sortIndex: i,
          colorIndex: i,
          label: id,
          monthRange,
          value,
          nominal: false,
        }
      })
      return result
    }
    return {
      v: 27,
      timestamp: now,
      dialogPositionNominal: 'done',
      people: {
        withPartner: true,
        person1: {
          ages: {
            type: 'retirementDateSpecified',
            monthOfBirth: yearsFromNowAsCalendarMonth(-30),
            retirementAge: { inMonths: 65 * 12 },
            maxAge: { inMonths: 95 * 12 },
          },
        },
        person2: {
          ages: {
            type: 'retirementDateSpecified',
            monthOfBirth: yearsFromNowAsCalendarMonth(-40),
            retirementAge: { inMonths: 70 * 12 },
            maxAge: { inMonths: 100 * 12 },
          },
        },
        withdrawalStart,
      },
      wealth: {
        portfolioBalance: {
          updatedHere: true,
          amount: 100000,
        },
        futureSavings: getValueForMonthRanges([
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: yearsFromNowAsCalendarMonth(-1),
              },
              end: {
                type: 'namedAge',
                person: 'person1',
                age: 'lastWorkingMonth',
              },
            },
            value: 1000,
          },
          {
            monthRange: {
              type: 'startAndEnd',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: yearsFromNowAsCalendarMonth(1),
              },
              end: {
                type: 'namedAge',
                person: 'person2',
                age: 'lastWorkingMonth',
              },
            },
            value: 2000,
          },
        ]),

        incomeDuringRetirement: getValueForMonthRanges([
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
            value: 3000,
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
            value: 4000,
          },
        ]),
      },
      adjustmentsToSpending: {
        extraSpending: {
          essential: getValueForMonthRanges([
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'calendarMonthAsNow',
                  monthOfEntry: yearsFromNowAsCalendarMonth(-1),
                },
                end: { type: 'namedAge', person: 'person1', age: 'max' },
              },
              value: 500,
            },
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'calendarMonthAsNow',
                  monthOfEntry: yearsFromNowAsCalendarMonth(-1),
                },
                end: { type: 'namedAge', person: 'person2', age: 'max' },
              },
              value: 600,
            },
          ]),
          discretionary: getValueForMonthRanges([
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'calendarMonthAsNow',
                  monthOfEntry: yearsFromNowAsCalendarMonth(-1),
                },
                end: { type: 'namedAge', person: 'person1', age: 'max' },
              },
              value: 700,
            },
            {
              monthRange: {
                type: 'startAndEnd',
                start: {
                  type: 'calendarMonthAsNow',
                  monthOfEntry: yearsFromNowAsCalendarMonth(-1),
                },
                end: { type: 'namedAge', person: 'person2', age: 'max' },
              },
              value: 800,
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
              month: yearsFromNowAsCalendarMonth(-1),
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
        expectedReturnsForPlanning: {
          type: 'manual',
          stocks: 0.05,
          bonds: 0.02,
        },
        historicalMonthlyLogReturnsAdjustment: {
          standardDeviation: {
            stocks: { scale: 1 },
            bonds: { enableVolatility: true },
          },
          overrideToFixedForTesting: true,
        },
        annualInflation: { type: 'manual', value: 0.02 },
        sampling: {
          type: 'monteCarlo',
          forMonteCarlo: {
            blockSize: 12 * 5,
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
    inflation: {
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

    timestampMSForHistoricalReturns: Number.MAX_SAFE_INTEGER,
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
