import _ from 'lodash'
import React, { useEffect, useRef, useState } from 'react'
import { extendPlanParams } from '../TPAWSimulator/PlanParamsExt'
import { processPlanParams } from '../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { TPAWRunInWorker } from '../TPAWSimulator/Worker/TPAWRunInWorker'
import { fGet } from '../Utils/Utils'
import { AppPage } from './App/AppPage'
import { useMarketData } from './App/WithMarketData'
import { AmountInput } from './Common/Inputs/AmountInput'

export const Perf = React.memo(() => {
  const [numRuns, setNumRuns] = useState(500)
  const [numPercentiles, setNumPercentiles] = useState(3)
  const percentiles = _.times(numPercentiles, (i) => i + 1)
  const marketData = useMarketData()
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
            const result = await fGet(workerRef.current).runSimulations(
              { canceled: false },
              numRuns,
              processPlanParams(params, marketData),
              percentiles,
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
            value={numRuns}
            onChange={setNumRuns}
            decimals={0}
            modalLabel="Number of Simulations"
          />
        </div>
        <div className="flex items-center justify-center">
          <h2 className=""># Percentiles:</h2>
          <AmountInput
            className="text-input mt-2"
            value={numPercentiles}
            onChange={setNumPercentiles}
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

const params = extendPlanParams({
  v: 19,
  warnedAbout14to15Converstion: true,
  warnedAbout16to17Converstion: true,
  dialogPosition: 'done',
  people: {
    withPartner: false,
    person1: {
      ages: {
        type: 'notRetired',
        currentMonth: 35 * 12,
        retirementMonth: 65 * 12,
        maxMonth: 100 * 12,
      },
    },
  },
  wealth: {
    currentPortfolioBalance: 0,
    futureSavings: [],
    retirementIncome: [],
  },
  adjustmentsToSpending: {
    tpawAndSPAW: {
      monthlySpendingCeiling: null,
      monthlySpendingFloor: null,
      legacy: {
        total: 0,
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
        at20: 12,
        deltaAtMaxAge: -2,
        forLegacyAsDeltaFromAt20: 2,
      },
      timePreference: 0,
      additionalAnnualSpendingTilt: 0,
    },
    tpawAndSPAW: {
      lmp: 0,
    },
    spaw: { annualSpendingTilt: 0.01 },
    spawAndSWR: {
      allocation: {
        start: { stocks: 0.5 },
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
      expected: { type: 'suggested' },
      historical: {
        type: 'adjusted',
        adjustment: { type: 'toExpected' },
        correctForBlockSampling: true,
      },
    },
    annualInflation: { type: 'suggested' },
    sampling: 'monteCarlo',
    samplingBlockSizeForMonteCarlo: 12 * 5,
    strategy: 'TPAW',
  },
  dev: {
    alwaysShowAllMonths: false,
  },
})
