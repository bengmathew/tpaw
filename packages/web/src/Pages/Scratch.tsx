import {
  PlanParams,
  currentPlanParamsVersion
} from '@tpaw/common'
import clsx from 'clsx'
import React from 'react'
import { normalizePlanParams } from '../Simulator/NormalizePlanParams/NormalizePlanParams'
import { CalendarDayFns } from '../Utils/CalendarDayFns'
import { WithWASM, useWASM } from './PlanRoot/PlanRootHelpers/WithWASM'

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
          const timestamp = 1693440000000
          const ianaTimezoneName = 'America/Los_Angeles'

          const planParamsNorm = normalizePlanParams(problemParams, {
            timestamp,
            calendarDay: CalendarDayFns.fromTimestamp(
              timestamp,
              ianaTimezoneName,
            ),
          })
        }}
      >
        run
      </button>
    </div>
  )
})

const problemMarketDataAtTime = {
  closingTime: 1693425600000,
  inflation: {
    closingTime: 1693425600000,
    value: 0.0226,
  },
  sp500: {
    closingTime: 1693425600000,
    value: 4514.8701,
  },
  bondRates: {
    closingTime: 1693425600000,
    fiveYear: 0.0211,
    sevenYear: 0.0196,
    tenYear: 0.0186,
    twentyYear: 0.0189,
    thirtyYear: 0.0197,
  },
  dailyStockMarketPerformance: {
    closingTime: 1693425600000,
    percentageChangeFromLastClose: {
      vt: 0.0024558095835295213,
      bnd: -0.00041873124432973823,
    },
  },
  timestampForMarketData: 1693440000000,
}

const problemParams: PlanParams = {
  v: currentPlanParamsVersion,
  timestamp: 1689083666832,
  dialogPositionNominal: 'done',
  people: {
    person1: {
      ages: {
        type: 'retiredWithNoRetirementDateSpecified',
        maxAge: {
          inMonths: 1200,
        },
        currentAgeInfo: {
          isDatedPlan: true,
          monthOfBirth: {
            year: 1947,
            month: 5,
          },
        },
      },
    },
    person2: {
      ages: {
        type: 'retiredWithNoRetirementDateSpecified',
        maxAge: {
          inMonths: 1200,
        },
        currentAgeInfo: {
          isDatedPlan: true,
          monthOfBirth: {
            year: 1947,
            month: 3,
          },
        },
      },
    },
    withPartner: true,
    withdrawalStart: 'person1',
  },
  wealth: {
    portfolioBalance: {
      updatedHere: true,
      amount: 3000000,
      isDatedPlan: true,
    },
    futureSavings: {},
    incomeDuringRetirement: {
      qlsqyrquuv: {
        label: 'Social Securiry',
        nominal: false,
        id: 'qlsqyrquuv',
        sortIndex: 0,
        colorIndex: 0,
        amountAndTiming: {
          type: 'recurring',
          monthRange: {
            type: 'startAndEnd',
            start: {
              type: 'now',
              monthOfEntry: {
                isDatedPlan: true,
                calendarMonth: {
                  year: 2023,
                  month: 7,
                },
              },
            },
            end: {
              age: 'max',
              type: 'namedAge',
              person: 'person1',
            },
          },
          everyXMonths: 1,
          baseAmount: 2615,
          delta: null,
        },
      },
    },
  },
  adjustmentsToSpending: {
    extraSpending: {
      essential: {},
      discretionary: {},
    },
    tpawAndSPAW: {
      monthlySpendingCeiling: null,
      monthlySpendingFloor: null,
      legacy: {
        total: 500000,
        external: {},
      },
    },
  },
  risk: {
    tpaw: {
      riskTolerance: {
        at20: 17,
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
          month: {
            type: 'now',
            monthOfEntry: {
              isDatedPlan: true,
              calendarMonth: {
                year: 2023,
                month: 6,
              },
            },
          },
          stocks: 0.5,
        },
        intermediate: {},
        end: {
          stocks: 0.5,
        },
      },
    },
    swr: {
      withdrawal: {
        type: 'asPercentPerYear',
        percentPerYear: 0.047,
      },
    },
  },
  advanced: {
    returnsStatsForPlanning: {
      expectedValue: {
        empiricalAnnualNonLog: {
          type: 'conservativeEstimate,20YearTIPSYield',
        },
      },
      standardDeviation: {
        stocks: {
          scale: {
            log: 1,
          },
        },
        bonds: {
          scale: {
            log: 0,
          },
        },
      },
    },
    historicalReturnsAdjustment: {
      standardDeviation: {
        bonds: {
          scale: {
            log: 1,
          },
        },
      },
      overrideToFixedForTesting: { type: 'none' },
    },
    sampling: {
      type: 'monteCarlo',
      data: {
        blockSize: {
          inMonths: 60,
        },
        staggerRunStarts: true,
      },
    },
    annualInflation: {
      type: 'suggested',
    },
    strategy: 'TPAW',
  },
  results: null,
  datingInfo: {
    isDated: true,
  },
}

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
      },
      overrideToFixedForTesting: { type: 'none' },
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
