import { PlanParams, getDefaultPlanParams, letIn } from '@tpaw/common'
import clsx from 'clsx'
import React from 'react'
import { extendPlanParams } from '../UseSimulator/ExtentPlanParams'
import {
  PlanParamsNormalized,
  normalizePlanParams,
} from '../UseSimulator/NormalizePlanParams'
import { WithWASM, useWASM } from './PlanRoot/PlanRootHelpers/WithWASM'
import * as Rust from '@tpaw/simulator'
import { nominalToReal } from '../Utils/NominalToReal'
import _ from 'lodash'
import { processPlanParams } from '../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { CallRust } from '../UseSimulator/PlanParamsProcessed/CallRust'
import { fWASM } from '../UseSimulator/Simulator/GetWASM'

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
          const planParamsExt = extendPlanParams(
            planParams,
            Date.now(),
            'America/Los_Angeles',
          )
          const nowAsCalendarMonth = letIn(
            planParamsExt.getZonedTime(Date.now()),
            (zonedTime) => ({
              year: zonedTime.year,
              month: zonedTime.month,
            }),
          )

          let start = performance.now()
          let n = 1000
          for (let i = 0; i < n; i++) {
            const planParamsNorm = normalizePlanParams(planParamsExt)
            // CallRust.processPlanParams(planParamsNorm, marketData)
            // const planParamsProcessed = processPlanParams(
            //   planParamsExt,
            //   1000000,
            //   marketData,
            // )
          }
          console.log('process_plan_params1', (performance.now() - start) / n)
          // start = performance.now()
          // let p = JSON.parse(planParamsProcessed.without_arrays())
          // let a1 = planParamsProcessed.array('historicalMonthlyLogReturnsAdjustedBonds').slice()
          // let a2 = planParamsProcessed.array('historicalMonthlyLogReturnsAdjustedStocks').slice()
          // console.log('process_plan_params2', performance.now() - start)
          // start = performance.now()
          // console.dir(p)
          // planParamsProcessed.free()
        }}
      >
        run
      </button>
    </div>
  )
})

const testPlanParams: PlanParams = {
  v: 27,
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
        value: 10000,
        nominal: false,
        sortIndex: 0,
        colorIndex: 1,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            type: 'calendarMonthAsNow',
            monthOfEntry: {
              year: 2023,
              month: 9,
            },
          },
        },
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
        value: 5000,
        nominal: false,
        sortIndex: 0,
        colorIndex: 0,
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
        blockSize: 60,
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
            value: 0,
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
          value: 2000,
          nominal: false,
          sortIndex: 1,
          colorIndex: 1,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 180,
          },
        },
        ibcehvmksf: {
          id: 'ibcehvmksf',
          label: 'Kaavini Activities',
          value: 800,
          nominal: false,
          sortIndex: 6,
          colorIndex: 6,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 180,
          },
        },
        lzykdntqcg: {
          id: 'lzykdntqcg',
          label: 'Theera School',
          value: 500,
          nominal: false,
          sortIndex: 2,
          colorIndex: 2,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 156,
          },
        },
        srjjqswpnk: {
          id: 'srjjqswpnk',
          label: 'Mortgage',
          value: 4000,
          nominal: false,
          sortIndex: 0,
          colorIndex: 0,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 360,
          },
        },
        uqavaxaxyg: {
          id: 'uqavaxaxyg',
          label: 'Kaavini School',
          value: 500,
          nominal: false,
          sortIndex: 4,
          colorIndex: 4,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonth',
              calendarMonth: {
                year: 2024,
                month: 9,
              },
            },
            numMonths: 168,
          },
        },
        vicjhlgzau: {
          id: 'vicjhlgzau',
          label: 'Theera Activities',
          value: 800,
          nominal: false,
          sortIndex: 5,
          colorIndex: 5,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 156,
          },
        },
        vkwtzracsk: {
          id: 'vkwtzracsk',
          label: 'Kaavini NWMS',
          value: 2000,
          nominal: false,
          sortIndex: 3,
          colorIndex: 3,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              type: 'calendarMonthAsNow',
              monthOfEntry: {
                year: 2023,
                month: 9,
              },
            },
            numMonths: 12,
          },
        },
      },
      discretionary: {},
    },
  },
  dialogPositionNominal: 'done',
}
