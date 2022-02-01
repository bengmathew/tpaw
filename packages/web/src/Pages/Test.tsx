import * as cookie from 'cookie'
import React, { useEffect, useState } from 'react'
import { runTPAWSimulation } from '../TPAWSimulator/RunTPAWSimulation'
import { TPAWParams } from '../TPAWSimulator/TPAWParams'
import { formatCurrency } from '../Utils/FormatCurrency'
import { Config } from './Config'

export const Test = React.memo(() => {
  if (Config.client.production) throw new Error()

  const [rows, setRows] = useState<string[][]>([])

  useEffect(() => {
    const simulationUsingExpectedReturns = runTPAWSimulation(
      testParams,
      null
    ).byYearFromNow

    const result = runTPAWSimulation(testParams, {
      simulationUsingExpectedReturns,
      randomIndexesIntoHistoricalReturnsByYear,
    })
    const delta = result.byYearFromNow
      .map(x => x.withdrawal)
      .map((x, i) => [
        `${i + testParams.age.start}`,
        (x - excel[i]).toFixed(10),
        `${formatCurrency(x)}`,
        `${x}`,
      ])
    // .slice(30, 31)
    setRows(delta)
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
  0, 0, 0, 0, 33139.87991, 34381.7998, 35535.58675, 109109.1077, 36988.32408,
  38554.71586, 39805.86501, 39628.63842, 40670.20002, 41865.05113, 43254.25183,
  40998.6241, 41552.68172, 41928.66716, 41714.11509, 39357.36416, 40132.59094,
  40882.33671, 40361.34975, 41019.51749, 41014.02493, 41754.84778, 40728.83522,
  43001.86675, 45720.40458, 46582.75394, 49506.09918, 51115.98513, 51928.55046,
  52824.3226, 53034.80522, 51248.0923, 49674.74647, 48321.7706, 51328.68388,
  49395.77456, 52523.4986, 52116.98104, 56273.75241, 57594.53796, 55459.31348,
  58327.86432, 58798.63632, 60198.88888, 57770.06358, 58308.72307,
]

const testParams: TPAWParams = {
  v: 3,
  age: {
    start: 25,
    retirement: 55,
    end: 100,
  },
  returns: {
    expected: {
      stocks: 0.035,
      bonds: 0.01,
    },
    historical: {
      adjust: {
        type: 'by',
        stocks: 0.048,
        bonds: 0.03,
      },
    },
  },
  inflation: 0.02,
  targetAllocation: {
    regularPortfolio: {stocks: 0.3},
    legacyPortfolio: {stocks: 0.7},
  },
  scheduledWithdrawalGrowthRate: 0.01,
  savingsAtStartOfStartYear: 50000,
  savings: [
    {
      label: 'Savings',
      yearRange: {start: 'start', end: 54},
      value: 25000,
      nominal: false,
    },
  ],
  retirementIncome: [
    {
      label: 'Social Security',
      yearRange: {start: 70, end: 'end'},
      value: 30000,
      nominal: false,
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
      {
        label: null,
        yearRange: {start: 58, end: 58},
        value: 100000,
        nominal: false,
      },
    ],
  },
  spendingCeiling: null,
  spendingFloor: null,
  legacy: {
    total: 300000,
    external: [],
  },
}

const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  [
    136, 24, 150, 98, 91, 53, 50, 5, 47, 111, 111, 64, 12, 121, 77, 61, 64, 41,
    24, 76, 79, 38, 116, 63, 106, 9, 108, 2, 87, 109, 5, 150, 25, 139, 55, 91,
    89, 19, 74, 91, 131, 18, 2, 25, 60, 110, 97, 48, 136, 41, 97, 145, 54, 112,
    118, 105, 91, 101, 102, 12, 17, 124, 145, 51, 64, 121, 108, 125, 146, 20, 7,
    90, 140, 60, 26, 53,
  ].map(x => x - 1)[year]
