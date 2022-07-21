import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {extendTPAWParams} from '../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {TPAWRunInWorker} from '../TPAWSimulator/Worker/TPAWRunInWorker'
import {fGet} from '../Utils/Utils'
import {AppPage} from './App/AppPage'

const numRuns = 50000
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export const Perf = React.memo(() => {
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
              params,
              percentiles
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

const params = processTPAWParams(
  extendTPAWParams({
    v: 12,
    strategy: 'TPAW',
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 25, retirement: 55, max: 100},
      },
    },
    returns: {
      expected: {type: 'suggested'},
      historical: {type: 'default', adjust: {type: 'toExpected'}},
    },
    inflation: {type: 'suggested'},
    targetAllocation: {
      regularPortfolio: {
        forTPAW: {
          start: {stocks: 0.35},
          intermediate: [],
          end: {stocks: 0.35},
        },
        forSPAWAndSWR: {
          start: {stocks: 0.5},
          intermediate: [],
          end: {stocks: 0.5},
        },
      },

      legacyPortfolio: {stocks: 0.7},
    },
    swrWithdrawal: {type: 'asPercent', percent: 0.04},
    scheduledWithdrawalGrowthRate: 0.02,
    savingsAtStartOfStartYear: 50000,
    savings: [
      {
        label: 'Savings',
        yearRange: {
          type: 'startAndEnd',
          start: {type: 'now'},
          end: {type: 'namedAge', person: 'person1', age: 'lastWorkingYear'},
        },
        value: 10000,
        nominal: false,
        id: 0,
      },
    ],
    retirementIncome: [
      {
        label: 'Social Security',
        yearRange: {
          type: 'startAndEnd',
          start: {type: 'numericAge', person: 'person1', age: 70},
          end: {type: 'namedAge', person: 'person1', age: 'max'},
        },
        value: 20000,
        nominal: false,
        id: 0,
      },
    ],
    withdrawals: {
      lmp: 0,
      essential: [
        {
          label: null,
          yearRange: {
            type: 'startAndEnd',
            start: {type: 'numericAge', person: 'person1', age: 76},
            end: {type: 'numericAge', person: 'person1', age: 76},
          },
          value: 100000,
          nominal: false,
          id: 0,
        },
      ],
      discretionary: [
        {
          label: null,
          yearRange: {
            type: 'startAndEnd',
            start: {type: 'numericAge', person: 'person1', age: 58},
            end: {type: 'numericAge', person: 'person1', age: 58},
          },
          value: 100000,
          nominal: false,
          id: 1,
        },
      ],
    },
    spendingCeiling: null,
    spendingFloor: null,
    legacy: {
      total: 50000,
      external: [],
    },
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  })
)
