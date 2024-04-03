import { PlanParams, currentPlanParamsVersion, letIn } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import clsx from 'clsx'
import React from 'react'
import { WithWASM, useWASM } from './PlanRoot/PlanRootHelpers/WithWASM'
import { normalizePlanParams } from '../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { fWASM } from '../UseSimulator/Simulator/GetWASM'
import { CallRust } from '../UseSimulator/PlanParamsProcessed/CallRust'

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
            inflation: {
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
            timestampMSForHistoricalReturns: Number.MAX_SAFE_INTEGER,
          }
          const planParams = testPlanParams
          const planParamsNorm = normalizePlanParams(planParams, {
            year: 2024,
            month: 4,
          })
          const planParamsRust = CallRust.getPlanParamsRust(planParamsNorm)
          const x =
            fWASM().process_market_data_for_expected_returns_for_planning_presets(
              planParamsRust.advanced.sampling,
              planParamsRust.advanced.historicalMonthlyLogReturnsAdjustment
                .standardDeviation,
              marketData,
            )
          console.dir(x)
        }}
      >
        run
      </button>
    </div>
  )
})

const testPlanParams: PlanParams = {
  v: currentPlanParamsVersion,
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
            year: 2023,
            month: 4,
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
        monthOfBirth: {
          year: 1983,
          month: 1,
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
        monthOfBirth: {
          year: 1982,
          month: 11,
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
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
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
      forMonteCarlo: {
        blockSize: { inMonths: 60 },
        staggerRunStarts: true,
      },
    },
    expectedReturnsForPlanning: {
      type: 'manual',
      bonds: 0.023,
      stocks: 0.049,
    },
    historicalMonthlyLogReturnsAdjustment: {
      standardDeviation: {
        stocks: {
          scale: 1.14,
        },
        bonds: {
          enableVolatility: true,
        },
      },

      overrideToFixedForTesting: false,
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
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
          label: 'Kaavini Activities',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 800,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
          label: 'Theera School',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 500,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
          label: 'Kaavini School',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 500,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
          label: 'Theera Activities',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 800,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
          label: 'Kaavini NWMS',
          amountAndTiming: {
            type: 'recurring',
            everyXMonths: 1,
            delta: null,
            baseAmount: 2000,
            monthRange: {
              type: 'startAndDuration',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 9,
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
