import { PlanParams, currentPlanParamsVersion } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import clsx from 'clsx'
import React from 'react'
import { normalizePlanParams } from '../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { CallRust } from '../UseSimulator/PlanParamsProcessed/CallRust'
import { processPlanParams } from '../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { WithWASM, useWASM } from './PlanRoot/PlanRootHelpers/WithWASM'
import { CalendarDayFns } from '../Utils/CalendarDayFns'

export const Scratch = React.memo(({ className }: { className?: string }) => {
  return (
    <WithWASM>
      <_Body />
    </WithWASM>
  )
})

export const _Body = React.memo(({ className }: { className?: string }) => {
  const { wasm } = useWASM()

  return (
    <div className={clsx(className)}>
      <button
        className=" btn2-lg btn2-dark rounded-full"
        onClick={() => {
          const marketData: Rust.DataForMarketBasedPlanParamValues = {
            closingTime: 1706648400000,
            inflation: {
              closingTime: 1706648400000,
              value: 0.0239,
            },
            sp500: {
              closingTime: 1706648400000,
              value: 4924.9702,
            },
            bondRates: {
              closingTime: 1706648400000,
              fiveYear: 0.0157,
              sevenYear: 0.0152,
              tenYear: 0.0149,
              twentyYear: 0.015,
              thirtyYear: 0.0154,
            },
            timestampForMarketData: Number.MAX_SAFE_INTEGER,
          }
          const planParams = testPlanParams
          const timestamp = planParams.timestamp + 1
          const planParamsNorm = normalizePlanParams(planParams, {
            timestamp,
            calendarDay: CalendarDayFns.fromTimestamp(
              timestamp,
              'America/Los_Angeles',
            ),
          })

          const planParamsRust = CallRust.getPlanParamsRust(planParamsNorm)
          let start = performance.now()

          CallRust.processPlanParams(planParamsNorm, marketData)
          console.log('fn0 took:', performance.now() - start)
          start = performance.now()

          processPlanParams(planParamsNorm, marketData)
          console.log('fn1 took:', performance.now() - start)
          start = performance.now()

          // const riskJS = planParamsProcessRisk(
          //   planParamsNorm,
          //   planParamsProcessed.returnsStatsForPlanning,
          // )
          // console.log('riskjs took:', performance.now() - start)
          // start = performance.now()

          // const data = result.get()
          // console.log('get bufffer took:', performance.now() - start)
          // start = performance.now()

          // const decoded = unpack(data)
          // console.dir(decoded)
          // console.log('decoding took:', performance.now() - start)
          // start = performance.now()

          // let decoder = avro
          //   .createBlobDecoder(new Blob([data]))
          //   .on('metadata', (data: any) => console.dir(data))
          //   .on('data', (data: any) => {
          //     console.log('decode took:', performance.now() - start)
          //     start = performance.now()
          //     console.dir(data)
          //   })
        }}
      >
        run
      </button>
    </div>
  )
})

const testPlanParams: PlanParams = {
  v: currentPlanParamsVersion,
  datingInfo: {
    isDated: true,
  },
  risk: {
    swr: {
      withdrawal: {
        type: 'asPercentPerYear',
        percentPerYear: 0.034,
      },
    },
    spaw: {
      annualSpendingTilt: 0.004,
    },
    tpaw: {
      riskTolerance: {
        at20: 13,
        deltaAtMaxAge: -2,
        forLegacyAsDeltaFromAt20: 2,
      },
      timePreference: 0,
      additionalAnnualSpendingTilt: 0,
    },
    spawAndSWR: {
      allocation: {
        end: {
          stocks: 0.5,
        },
        start: {
          month: {
            type: 'now',
            monthOfEntry: {
              isDatedPlan: true,
              calendarMonth: { year: 2023, month: 4 },
            },
          },
          stocks: 0.5,
        },
        intermediate: {},
      },
    },
    tpawAndSPAW: {
      lmp: 0,
    },
  },
  people: {
    person1: {
      ages: {
        type: 'retirementDateSpecified',
        maxAge: {
          inMonths: 1236,
        },
        currentAgeInfo: {
          isDatedPlan: true,
          monthOfBirth: { year: 1983, month: 1 },
        },
        retirementAge: {
          inMonths: 780,
        },
      },
    },
    person2: {
      ages: {
        type: 'retirementDateSpecified',
        maxAge: {
          inMonths: 1200,
        },
        currentAgeInfo: {
          isDatedPlan: true,
          monthOfBirth: { year: 1982, month: 11 },
        },
        retirementAge: {
          inMonths: 780,
        },
      },
    },
    withPartner: true,
    withdrawalStart: 'person1',
  },
  wealth: {
    futureSavings: {
      feebiphizs: {
        id: 'feebiphizs',
        label: 'Total',
        amountAndTiming: {
          type: 'recurring',
          everyXMonths: 1,
          delta: null,
          baseAmount: 10000,
          monthRange: {
            type: 'startAndEnd',
            end: {
              age: 'lastWorkingMonth',
              type: 'namedAge',
              person: 'person1',
            },
            start: {
              type: 'now',
              monthOfEntry: {
                isDatedPlan: true,
                calendarMonth: {
                  year: 2023,
                  month: 9,
                },
              },
            },
          },
        },
        nominal: false,
        sortIndex: 0,
        colorIndex: 1,
      },
    },
    portfolioBalance: {
      isDatedPlan: true,
      updatedHere: true,
      amount: 900000,
    },
    incomeDuringRetirement: {
      hrzsmdujpd: {
        id: 'hrzsmdujpd',
        label: 'Social Security',
        amountAndTiming: {
          type: 'recurring',
          everyXMonths: 1,
          delta: null,
          baseAmount: 5000,
          monthRange: {
            end: {
              age: 'max',
              type: 'namedAge',
              person: 'person1',
            },
            type: 'startAndEnd',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
          },
        },
        nominal: false,
        sortIndex: 0,
        colorIndex: 0,
      },
    },
  },
  results: {
    displayedAssetAllocation: {
      stocks: 0.5557,
    },
  },
  advanced: {
    strategy: 'TPAW',
    sampling: {
      type: 'monteCarlo',
      data: {
        blockSize: { inMonths: 60 },
        staggerRunStarts: true,
      },
    },
    returnsStatsForPlanning: {
      expectedValue: {
        empiricalAnnualNonLog: {
          type: 'fixed',
          bonds: 0.023,
          stocks: 0.049,
        },
      },
      standardDeviation: {
        stocks: { scale: { log: 1.14 } },
        bonds: { scale: { log: 0 } },
      },
    },
    historicalReturnsAdjustment: {
      standardDeviation: {
        bonds: { scale: { log: 1 } },
        overrideToFixedForTesting: false,
      },
    },
    annualInflation: {
      type: 'suggested',
    },
  },
  timestamp: 1708191348891,
  adjustmentsToSpending: {
    tpawAndSPAW: {
      legacy: {
        total: 0,
        external: {
          eunijmppod: {
            id: 'eunijmppod',
            label: null,
            amount: 0,
            nominal: false,
            sortIndex: 0,
            colorIndex: 0,
          },
        },
      },
      monthlySpendingFloor: null,
      monthlySpendingCeiling: null,
    },
    extraSpending: {
      essential: {
        fvpxstjfml: {
          id: 'fvpxstjfml',
          label: 'Kids College',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 2000,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 180 },
            },
          },
          nominal: false,
          sortIndex: 1,
          colorIndex: 1,
        },
        ibcehvmksf: {
          id: 'ibcehvmksf',
          label: 'Child1 Activities',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 800,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 180 },
            },
          },
          nominal: false,
          sortIndex: 6,
          colorIndex: 6,
        },
        lzykdntqcg: {
          id: 'lzykdntqcg',
          label: 'Child2 School',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 500,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 156 },
            },
          },
          nominal: false,
          sortIndex: 2,
          colorIndex: 2,
        },
        srjjqswpnk: {
          id: 'srjjqswpnk',
          label: 'Mortgage',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 4000,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 360 },
            },
          },
          nominal: false,
          sortIndex: 0,
          colorIndex: 0,
        },
        uqavaxaxyg: {
          id: 'uqavaxaxyg',
          label: 'Child1 School',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 500,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 156 },
            },
          },
          nominal: false,
          sortIndex: 4,
          colorIndex: 4,
        },
        vicjhlgzau: {
          id: 'vicjhlgzau',
          label: 'Child2 Activities',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 800,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 156 },
            },
          },
          nominal: false,
          sortIndex: 5,
          colorIndex: 5,
        },
        vkwtzracsk: {
          id: 'vkwtzracsk',
          label: 'Child1 NWMS',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 2000,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 9 },
                },
              },
              duration: { inMonths: 12 },
            },
          },
          nominal: false,
          sortIndex: 3,
          colorIndex: 3,
        },
      },
      discretionary: {},
    },
  },
  dialogPositionNominal: 'done',
}
