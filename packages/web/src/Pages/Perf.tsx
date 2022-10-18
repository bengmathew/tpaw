import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {extendTPAWParams} from '../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {TPAWRunInWorker} from '../TPAWSimulator/Worker/TPAWRunInWorker'
import {fGet} from '../Utils/Utils'
import {AppPage} from './App/AppPage'
import {useMarketData} from './App/WithMarketData'

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export const Perf = React.memo(() => {
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
              {canceled: false},
              numRuns,
              processTPAWParams(params, marketData),
              percentiles,
              marketData
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
        <div className="grid" style={{grid: 'auto / auto'}}>
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

const params = extendTPAWParams({
  v: 14,
  strategy: 'TPAW',
  dialogMode: true,
  people: {
    withPartner: false,
    person1: {
      displayName: null,
      ages: {type: 'notRetired', current: 35, retirement: 65, max: 100},
    },
  },
  currentPortfolioBalance: 0,
  futureSavings: [],
  retirementIncome: [],
  extraSpending: {
    essential: [],
    discretionary: [],
  },
  legacy: {
    tpawAndSPAW: {
      total: 0,
      external: [],
    },
  },
  risk: {
    useTPAWPreset: false,
    tpawPreset: 'riskLevel-2',
    customTPAWPreset: null,
    savedTPAWPreset: null,
    tpaw: {
      allocation: {
        start: {stocks: 0.4},
        intermediate: [],
        end: {stocks: 0.3},
      },
      allocationForLegacy: {stocks: 0.7},
    },
    tpawAndSPAW: {
      spendingCeiling: null,
      spendingFloor: null,
      spendingTilt: 0.01,
      lmp: 0,
    },
    spawAndSWR: {
      allocation: {
        start: {stocks: 0.5},
        intermediate: [],
        end: {stocks: 0.5},
      },
    },
    swr: {
      withdrawal: {type: 'default'},
    },
  },

  returns: {
    expected: {type: 'suggested'},
    historical: {type: 'default', adjust: {type: 'toExpected'}},
  },
  inflation: {type: 'suggested'},
  sampling: 'monteCarlo',
  display: {
    alwaysShowAllYears: false,
  },
})
