import {
  PlanParams,
  PlanParamsChangeActionCurrent,
  ValueForMonthRange,
  assert,
  block,
  fGet,
  generateSmallId,
  getSlug,
  letIn,
} from '@tpaw/common'
import jsonpatch from 'fast-json-patch'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { env } from 'process'
import * as uuid from 'uuid'
import { Clients } from '../../../../Clients.js'
import { cloneJSON } from '../../../../Utils/CloneJSON.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan
  .command('createTestPlan <emailOrId> <actionsPerDay> <numDays>')
  .action(
    async (emailOrId: string, actionsPerDayStr: string, numDaysStr: string) => {
      assert(env['NODE_ENV'] === 'development')

      const actionsPerDay = parseInt(actionsPerDayStr)
      assert(!isNaN(actionsPerDay))
      const numDays = parseInt(numDaysStr)
      assert(!isNaN(numDays))

      const email = emailOrId.includes('@')
        ? emailOrId
        : fGet((await Clients.firebaseAuth.getUser(emailOrId)).email)
      const userId = fGet(await Clients.firebaseAuth.getUserByEmail(email)).uid

      const msPerDay = 1000 * 60 * 60 * 24

      const timestamps = block(() => {
        const fromNow = letIn({ curr: Date.now() }, (vars) =>
          _.range(actionsPerDay * numDays).map((i) => {
            vars.curr = vars.curr + (i % actionsPerDay === 0 ? msPerDay : 60)
            return vars.curr
          }),
        )
        const offset = fGet(_.last(fromNow)) - Date.now() + 1000 * 60
        return fromNow.map((x) => x - offset)
      })

      const initTimestamp = fGet(timestamps[0]) - 1000 * 60
      let currParams = startingParams(initTimestamp)
      const paramsChangeHistory: ReturnType<
        typeof _applyChange
      >['historyItem'][] = [
        {
          planParamsChangeId: uuid.v4(),
          timestamp: new Date(initTimestamp),
          reverseDiff: [],
          change: { type: 'startCopiedFromBeforeHistory', value: null },
        },
        ...timestamps.map((timestamp) => {
          assert(currParams.wealth.portfolioBalance.updatedHere)
          const { historyItem, params } = _applyChange(
            currParams.wealth.portfolioBalance.amount + 1,
            timestamp,
            currParams,
          )
          currParams = params
          return historyItem
        }),
      ]

      const currPlans = await Clients.prisma.planWithHistory.findMany({
        where: { userId },
      })

      const label = `Test ${actionsPerDay}x${numDays}`
      const now = new Date()
      await Clients.prisma.user.update({
        where: { id: userId },
        data: {
          planWithHistory: {
            create: {
              planId: uuid.v4(),
              isMain: false,
              label,
              slug: getSlug(
                label,
                currPlans.map((x) => x.slug),
              ),
              addedToServerAt: now,
              sortTime: now,
              lastSyncAt: now,
              resetCount: 0,
              endingParams: currParams,
              paramsChangeHistory: {
                createMany: {
                  data: paramsChangeHistory,
                },
              },
              reverseHeadIndex: 0,
            },
          },
        },
      })
    },
  )

const _applyChange = (value: number, timestamp: number, params: PlanParams) => {
  const clone = cloneJSON(params)
  clone.timestamp = timestamp
  clone.wealth.portfolioBalance = { updatedHere: true, amount: value }
  clone.results = { displayedAssetAllocation: { stocks: 0.5 } }

  const reverseDiff = jsonpatch.compare(clone, params)
  const change: PlanParamsChangeActionCurrent = {
    type: 'setCurrentPortfolioBalance',
    value,
  } as PlanParamsChangeActionCurrent
  return {
    historyItem: {
      planParamsChangeId: uuid.v4(),
      timestamp: new Date(timestamp),
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
      reverseDiff: reverseDiff as any,
      change,
    },
    params: clone,
  }
}

const startingParams = (timestamp: number): PlanParams => ({
  v: 27,
  risk: {
    swr: {
      withdrawal: {
        type: 'default',
      },
    },
    spaw: {
      annualSpendingTilt: 0.008,
    },
    tpaw: {
      riskTolerance: {
        at20: 12,
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
            year: DateTime.fromMillis(timestamp).year,
            month: DateTime.fromMillis(timestamp).month,
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
          inMonths: 1200,
        },
        monthOfBirth: {
          year: 1988,
          month: 5,
        },
        retirementAge: {
          inMonths: 780,
        },
      },
    },
    withPartner: false,
  },
  wealth: {
    futureSavings: block(() => {
      const id1 = generateSmallId()
      const id2 = generateSmallId()
      const result: Record<string, ValueForMonthRange> = {}
      result[id1] = {
        id: id1,
        sortIndex: 0,
        colorIndex: 0,
        label: 'From My Salary',
        value: 300,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'endAndNumMonths',
          numMonths: 12,
        },
      }

      result[id2] = {
        id: id2,
        sortIndex: 1,
        colorIndex: 1,
        label: 'From Spouse Salary',
        value: 0,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'endAndNumMonths',
          numMonths: 12,
        },
      }
      return result
    }),
    portfolioBalance: {
      updatedHere: true,
      amount: 22000,
    },
    incomeDuringRetirement: block(() => {
      const id1 = generateSmallId()
      const id2 = generateSmallId()
      const id3 = generateSmallId()
      const result: Record<string, ValueForMonthRange> = {}
      result[id1] = {
        id: id1,
        sortIndex: 0,
        colorIndex: 2,
        label: 'Rental property',
        value: 500,
        nominal: false,
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
      }
      result[id2] = {
        id: id2,
        sortIndex: 1,
        colorIndex: 3,
        label: 'Social Security',
        value: 200,
        nominal: false,
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
      }
      result[id3] = {
        id: id3,
        sortIndex: 2,
        colorIndex: 4,
        label: 'Spouse Social Security',
        value: 0,
        nominal: false,
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
      }
      return result
    }),
  },
  advanced: {
    sampling: {
      type: 'monteCarlo',
      forMonteCarlo: {
        blockSize: 12 * 5,
        staggerRunStarts: true,
      },
    },
    strategy: 'TPAW',
    expectedReturnsForPlanning: {
      type: 'regressionPrediction,20YearTIPSYield',
    },
    historicalMonthlyLogReturnsAdjustment: {
      standardDeviation: {
        stocks: { scale: 1 },
        bonds: { enableVolatility: true },
      },
      overrideToFixedForTesting:false,
    },
    annualInflation: {
      type: 'suggested',
    },
  },
  timestamp,
  dialogPositionNominal: 'done',
  adjustmentsToSpending: {
    tpawAndSPAW: {
      legacy: {
        total: 0,
        external: {},
      },
      monthlySpendingFloor: 1000,
      monthlySpendingCeiling: 7000,
    },
    extraSpending: {
      essential: block(() => {
        const id1 = generateSmallId()
        const result: Record<string, ValueForMonthRange> = {}
        result[id1] = {
          id: id1,
          sortIndex: 0,
          colorIndex: 0,
          label: 'Travel',
          value: 0,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        }
        return result
      }),
      discretionary: block(() => {
        const id1 = generateSmallId()
        const result: Record<string, ValueForMonthRange> = {}
        result[id1] = {
          id: id1,
          sortIndex: 0,
          colorIndex: 1,
          label: 'Vacation',
          value: 100,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        }
        return result
      }),
    },
  },
  results: null,
})
