import React, {useEffect, useState} from 'react'
import {runTPAWSimulation} from '../TPAWSimulator/RunTPAWSimulation'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {formatCurrency} from '../Utils/FormatCurrency'
import {Config} from './Config'

export const Test = React.memo(() => {
  if (Config.client.production) throw new Error()

  const [rows, setRows] = useState<string[][]>([])

  useEffect(() => {
    const params = testParams
    const resultsFromUsingExpectedReturns = runTPAWSimulation({
      type: 'useExpectedReturns',
      params,
    }).byYearFromNow

    const result = runTPAWSimulation({
      type: 'useHistoricalReturns',
      params,
      resultsFromUsingExpectedReturns,
      randomIndexesIntoHistoricalReturnsByYear,
    })
    const delta = result.byYearFromNow
      .map(x => x.withdrawalAchieved.total)
      .map((x, i) => [
        `${i + params.people.person1.ages.current}`,
        `${x - excel[i]}`,
        `${formatCurrency(x)}`,
        `${x}`,
      ])
    setRows(delta)
    console.dir(result.byYearFromNow[30])
  }, [])

  return (
    <div className="">
      <div
        className="grid gap-x-8 "
        style={{
          grid: `auto / repeat(${
            rows.length === 0 ? 1 : rows[0].length
          }, auto) 1fr`,
        }}
      >
        {rows.map((row, i) => (
          <>
            <h2 key={i} className="font-mono">
              {i}
            </h2>
            {row.map((col, j) => (
              <h2 key={`${i}-${j}`} className=" font-mono ">
                {col}
              </h2>
            ))}
          </>
        ))}
      </div>
    </div>
  )
})

const excel = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 33996.97948, 36515.23335, 40819.66427, 39047.23173, 43282.84372,
  38642.9173, 38319.86888, 41266.22232, 40328.67974, 41360.64665, 42276.505,
  43892.50694, 47211.72622, 46141.98058, 47080.35786, 47697.01953, 50255.33648,
  50230.94697, 56982.40524, 60566.23041, 57757.97577, 55540.16595, 53381.92747,
  53425.15133, 59193.47445, 63332.54282, 65104.49289, 71132.33506, 71657.30804,
  74150.14793, 76264.39493, 84537.21839, 77955.49397, 85801.11707, 80446.64488,
  84921.79502, 94142.43829, 97608.06911, 101613.1702, 91215.93683, 94060.64039,
  101020.0051, 94153.9511, 87748.30235, 81002.60827, 79256.98725,
]

const testParams = processTPAWParams({
  v: 5,
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
    regularPortfolio: {stocks: 0.3},
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
        end: {type: 'numericAge', person: 'person1', age: 54},
      },
      value: 25000,
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
      value: 30000,
      nominal: false,
      id: 0,
    },
  ],
  withdrawals: {
    fundedByBonds: [
      // {
      //   label: null,
      //   yearRange: {start: 76, end: 76},
      //   value: 100000,
      //   nominal: false,
      // },
    ],
    fundedByRiskPortfolio: [
      // {
      //   label: null,
      //   yearRange: {start: 58, end: 58},
      //   value: 100000,
      //   nominal: false,
      //   id: 1,
      // },
    ],
  },
  spendingCeiling: null,
  spendingFloor: null,
  legacy: {
    total: 0,
    external: [],
  },
})
const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  [
    103, 34, 47, 58, 62, 105, 115, 6, 101, 60, 111, 146, 62, 21, 73, 122, 87,
    48, 52, 31, 136, 122, 129, 60, 9, 124, 56, 35, 108, 102, 28, 8, 20, 115, 71,
    137, 54, 120, 16, 114, 74, 149, 120, 114, 97, 79, 98, 112, 4, 49, 64, 49,
    89, 8, 34, 82, 75, 69, 74, 101, 121, 107, 127, 46, 68, 57, 91, 23, 67, 40,
    28, 37, 37, 107, 43, 38,
  ].map(x => x - 1)[year]
