import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {TPAWRunInWorker} from '../TPAWSimulator/Worker/TPAWRunInWorker'
import {fGet} from '../Utils/Utils'
import {AppPage} from './App/AppPage'

const numRuns = 500
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
    <AppPage title="Performance - TPAW Planner" curr="other">
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
            const lines = [
              `numCores: ${navigator.hardwareConcurrency || 4}`,
              ...fGet(result).perf.map(x => JSON.stringify(x)),
              '*',
              ...fGet(result).perfByYearsFromNow.map(x => JSON.stringify(x)),
              '*',
              ...fGet(result).perfByWorker.map(x => JSON.stringify(x)),
            ]
            console.dir('---------------')
            lines.forEach(x => console.dir(x))
            setResult(lines)
          }}
        >
          Test Performance
        </button>
        <div>
          {result.map((line, i) => (
            <h2 key={i} className="">
              {line}
            </h2>
          ))}
        </div>
      </div>
    </AppPage>
  )
})

const params = processTPAWParams({
  v: 6,
  strategy: 'SPAW',
  people: {
    withPartner: false,
    person1: {
      displayName: null,
      ages: {type: 'notRetired', current: 25, retirement: 55, max: 100},
    },
  },
  returns: {
    expected: {stocks: 0.035, bonds: 0.01},
    historical: {adjust: {type: 'by', stocks: 0.048, bonds: 0.03}},
  },
  inflation: 0.02,
  targetAllocation: {
    regularPortfolio: {
      forTPAW: {stocks: 0.35},
      forSPAW: {
        start: {stocks: 0.5},
        intermediate: [],
        end: {stocks: 0.5},
      },
    },

    legacyPortfolio: {stocks: 0.7},
  },
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
    lmp:0,
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
})
