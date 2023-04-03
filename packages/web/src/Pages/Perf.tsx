import { assert, calendarMonthFromTime } from '@tpaw/common'
import { DateTime } from 'luxon'
import React, { useEffect, useRef, useState } from 'react'
import { extendParams } from '../TPAWSimulator/ExtentParams'
import { processPlanParams } from '../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { TPAWRunInWorker } from '../TPAWSimulator/Worker/TPAWRunInWorker'
import { fGet } from '../Utils/Utils'
import { AppPage } from './App/AppPage'
import { useMarketData } from './App/WithMarketData'
import { AmountInput } from './Common/Inputs/AmountInput'

export const Perf = React.memo(() => {
  const [numOfSimulations, setNumOfSimulations] = useState(500)
  const marketData = useMarketData()
  const [currentTime] = useState(DateTime.local())
  const workerRef = useRef<TPAWRunInWorker | null>(null)
  useEffect(() => {
    workerRef.current = new TPAWRunInWorker()
    return () => workerRef.current?.terminate()
  }, [])

  const [result, setResult] = useState([''])

  return (
    <AppPage
      className="min-h-screen"
      title="Performance - TPAW Planner"
      curr="other"
    >
      <div className="h-screen flex flex-col justify-center items-center">
        <button
          className="rounded-full text-xl px-6 py-2 border-2 border-gray-700"
          onClick={async () => {
            const params = getParams(currentTime, numOfSimulations)
            assert(params.params.plan.wealth.portfolioBalance.isLastPlanChange)
            const result = await fGet(workerRef.current).runSimulations(
              { canceled: false },
              processPlanParams(
                params,
                params.params.plan.wealth.portfolioBalance.amount,
                marketData.latest,
              ),
              params,
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
          }}
        >
          Test Performance
        </button>
        <div className="flex items-center justify-center">
          <h2 className=""># Runs: </h2>
          <AmountInput
            className="text-input mt-2"
            value={numOfSimulations}
            onChange={setNumOfSimulations}
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

const getParams = (currentTime: DateTime, numOfSimulations: number) =>
  extendParams(
    {
      v: 20,
      plan: {
        timestamp: currentTime.valueOf(),
        dialogPosition: 'done',
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
            isLastPlanChange: true,
            amount: 100000,
            timestamp: currentTime.valueOf(),
          },
          futureSavings: [],
          retirementIncome: [],
        },
        adjustmentsToSpending: {
          tpawAndSPAW: {
            monthlySpendingCeiling: null,
            monthlySpendingFloor: null,
            legacy: {
              total: 1000000000,
              external: [],
            },
          },
          extraSpending: {
            essential: [],
            discretionary: [],
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
              intermediate: [],
              end: { stocks: 0.5 },
            },
          },
          swr: {
            withdrawal: { type: 'default' },
          },
        },

        advanced: {
          annualReturns: {
            expected: { type: 'manual', stocks: 0.04, bonds: 0.02 },
            // historical: {
            //   type: 'adjusted',
            //   adjustment: { type: 'toExpected' },
            //   correctForBlockSampling: true,
            // },
            historical: { type: 'unadjusted' },
          },
          annualInflation: { type: 'manual', value: 0.02 },
          sampling: 'monteCarlo',
          monteCarloSampling: {
            blockSize: 12 * 5,
            numOfSimulations,
          },

          strategy: 'TPAW',
        },
      },
      nonPlan: {
        migrationWarnings: {
          v14tov15: false,
          v16tov17: false,
          v19tov20: false,
        },
        percentileRange: { start: 5, end: 95 },
        defaultTimezone: {
          type: 'auto',
          ianaTimezoneName: currentTime.zoneName,
        },
        dev: {
          alwaysShowAllMonths: false,
          currentTimeFastForward: { shouldFastForward: false },
        },
      },
    },
    currentTime,
  )
