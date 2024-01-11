import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  NonPlanParams,
  assert,
  calendarMonthFromTime,
  currentPlanParamsVersion,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useRef, useState } from 'react'
import { extendPlanParams } from '../UseSimulator/ExtentPlanParams'
import { processPlanParams } from '../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { Simulator } from '../UseSimulator/Simulator/Simulator'
import { getSimulatorSingleton } from '../UseSimulator/UseSimulator'
import { fGet } from '../Utils/Utils'
import { AppPage } from './App/AppPage'
import { getMarketDataForTime } from './Common/GetMarketData'
import { AmountInput } from './Common/Inputs/AmountInput'
import { useMarketData } from './PlanRoot/PlanRootHelpers/WithMarketData'

export const Perf = React.memo(() => {
  const { marketData } = useMarketData()
  const [currentTime] = useState(DateTime.local())
  const [nonPlanParams, setNonPlanParams] = useState<NonPlanParams>(() =>
    getDefaultNonPlanParams(currentTime.toMillis()),
  )
  const workerRef = useRef<Simulator>(getSimulatorSingleton())

  const [result, setResult] = useState([''])

  return (
    <AppPage className="min-h-screen" title="Performance - TPAW Planner">
      <div className="h-screen flex flex-col justify-center items-center">
        <button
          className="rounded-full text-xl px-6 py-2 border-2 border-gray-700"
          onClick={() => {
            void (async () => {
              const planParamsExt = getParams(currentTime)
              const { planParams } = planParamsExt
              assert(planParams.wealth.portfolioBalance.updatedHere)
              const currMarketData = getMarketDataForTime(
                currentTime.toMillis(),
                marketData,
              )
              const planParamsProcessed = processPlanParams(
                planParamsExt,
                planParams.wealth.portfolioBalance.amount,
                currMarketData,
              )
              const result = await fGet(workerRef.current).runSimulations(
                { canceled: false },
                {
                  planParams,
                  planParamsProcessed,
                  planParamsExt,
                  numOfSimulationForMonteCarloSampling:
                    nonPlanParams.numOfSimulationForMonteCarloSampling,
                  randomSeed: DEFAULT_MONTE_CARLO_SIMULATION_SEED,
                },
              )

              const toLine = ([label, amount]: [string, number]) =>
                `${label}: ${amount.toFixed(2).padStart(8, ' ')}`
              const {
                main,
                sortAndPickPercentilesYearly,
                slowestSimulationWorker,
              } = fGet(result).perf
              const lines = [
                '------------------------------------',
                'NET',
                '------------------------------------',
                ...main.map(toLine),
                '------------------------------------',
                'SORT AND PICK PERCENTILES FOR YEARLY ',
                '------------------------------------',
                ...sortAndPickPercentilesYearly.map(toLine),
                '------------------------------------',
                'SLOWEST SIMULATION WORKER',
                '------------------------------------',
                ...slowestSimulationWorker.map(toLine),
              ]
              setResult(lines)
            })()
          }}
        >
          Test Performance
        </button>
        <div className="flex items-center justify-center">
          <h2 className=""># Runs: </h2>
          <AmountInput
            className="text-input mt-2"
            value={nonPlanParams.numOfSimulationForMonteCarloSampling}
            onChange={(x) => {
              setNonPlanParams((prev) => {
                const clone = _.cloneDeep(prev)
                clone.numOfSimulationForMonteCarloSampling = x
                return clone
              })
            }}
            decimals={0}
            modalLabel="Number of Simulations"
          />
        </div>

        <div className="grid" style={{ grid: 'auto / auto' }}>
          {result.map((line, i) => {
            const cols = line.split(':')
            return cols.map((col, r) => (
              <h2
                key={`${i}-${r}`}
                className={` font-mono
                ${cols.length === 1 ? ' col-span-2' : ''}
                ${r === 1 ? ' text-right' : 'font-mono'}
                `}
              >
                {col}
              </h2>
            ))
          })}
        </div>
      </div>
    </AppPage>
  )
})

const getParams = (currentTime: DateTime) =>
  extendPlanParams(
    {
      v: currentPlanParamsVersion,
      results: null,
      timestamp: currentTime.valueOf(),
      dialogPositionNominal: 'done',
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
          updatedHere: true,
          amount: 100000,
        },
        futureSavings: {},
        incomeDuringRetirement: {},
      },
      adjustmentsToSpending: {
        tpawAndSPAW: {
          monthlySpendingCeiling: null,
          monthlySpendingFloor: null,
          legacy: {
            total: 1000000000,
            external: {},
          },
        },
        extraSpending: {
          essential: {},
          discretionary: {},
        },
      },
      risk: {
        tpaw: {
          riskTolerance: {
            at20: 0,
            deltaAtMaxAge: 0,
            forLegacyAsDeltaFromAt20: 0,
          },
          timePreference: 0,
          additionalAnnualSpendingTilt: 0,
        },
        tpawAndSPAW: {
          lmp: 0,
        },
        spaw: { annualSpendingTilt: 0.0 },
        spawAndSWR: {
          allocation: {
            start: {
              month: calendarMonthFromTime(currentTime),
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
        expectedAnnualReturnForPlanning: {
          type: 'manual',
          stocks: 0.04,
          bonds: 0.02,
        },
        historicalReturnsAdjustment: {
          stocks: {
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
            volatilityScale: 1,
          },
          bonds: {
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
            enableVolatility: true,
          },
        },
        annualInflation: { type: 'manual', value: 0.02 },
        sampling: {
          type: 'monteCarlo',
          blockSizeForMonteCarloSampling: 12 * 5,
        },
        strategy: 'TPAW',
      },
    },
    currentTime.toMillis(),
    fGet(currentTime.zoneName),
  )
