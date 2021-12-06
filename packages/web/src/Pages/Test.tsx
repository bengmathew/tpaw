import React, { useEffect, useState } from 'react'
import { historicalReturns } from '../TPAWSimulator/HistoricalReturns'
import { runTPAWSimulation } from '../TPAWSimulator/RunTPAWSimulation'
import { TPAWParams } from '../TPAWSimulator/TPAWParams'

export const Test = React.memo(() => {
  const [rows, setRows] = useState<string[][]>([])
  useEffect(() => {
    const result = runTPAWSimulation(testParams, {
      randomIndexesIntoHistoricalReturnsByYear,
    })
    const delta = result.byYearFromNow
      .map(x => x.savingsPortfolioEndingBalance)
      .map((x, i) => [(x - excel[i]).toFixed(10), `${x}`])
    setRows(delta)
  }, [])

  return (
    <div className="">
      <div className="grid gap-x-8 " style={{grid: 'auto / auto  auto 1fr'}}>
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
  71410.3405525538, 127214.866368676, 154156.334651861, 232746.403065235,
  294970.159359571, 401760.296434754, 378722.77010794, 499844.846659743,
  482077.162089081, 543715.103045394, 609174.927242534, 633637.962688157,
  733833.817056491, 614362.524186434, 613003.013728942, 689831.993382394,
  729726.090244475, 926987.656287793, 994633.819892017, 850544.802864265,
  855371.99654033, 705910.967857607, 732837.203741971, 691313.505302104,
  696598.018233305, 770145.582972668, 1035015.90596952, 899343.191186578,
  858340.317422533, 633264.225713085, 551448.819210936, 545704.247481947,
  454142.814144108, 507995.482522703, 545103.851113852, 470601.791480639,
  484531.571537079, 485126.245753892, 424265.098055474, 390414.76814744,
  328630.147739464, 300778.345051793, 294220.034914258, 264590.951212112,
  156759.112942367, 135597.146097273, 130175.699912768, 142426.277441462,
  126722.344557313, 141849.92583054, 127312.858166529, 35545.8531675739,
  35881.6646549745, 30250.4521451013, 34551.9172208008, 13803.2883070527,
  13330.1701447789, 15930.7547665422, 18328.4197705491, 21423.7468432238,
  21709.9094741133, 21242.1296164349, 23584.8859223295, 21115.3989963479,
  17041.1026260935, 15273.5067469576, 14592.7756329563, 16626.0507292611,
  18681.0103009536, 13995.1037667366, 15759.823257704, 18501.0551084068,
  13980.0657880036, 14557.2813076232, 18983.8567519759, 19091.5902693105,
]

const testParams: TPAWParams = {
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
    regularPortfolio: {stocks: 0.35},
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
    {
      label: 'Social Security',
      yearRange: {start: 70, end: 'end'},
      value: 30000,
      nominal: false,
    },
  ],
  withdrawals: {
    fundedByBonds: [
      {
        label: null,
        yearRange: {start: 76, end: 76},
        value: 100000,
        nominal: false,
      },
    ],
    fundedByRiskPortfolio: [
      {
        label: null,
        yearRange: {start: 80, end: 80},
        value: 100000,
        nominal: false,
      },
    ],
  },
  spendingCeiling: null,
  legacy: {
    total: 300000,
    external: [],
  },
}

const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  [
    42, 88, 26, 75, 73, 125, 77, 52, 36, 129, 118, 41, 93, 132, 14, 4, 26, 15,
    18, 37, 108, 71, 41, 59, 39, 19, 65, 29, 59, 47, 148, 101, 64, 21, 126, 145,
    74, 5, 32, 122, 92, 98, 142, 106, 104, 96, 42, 102, 11, 150, 44, 139, 53,
    111, 113, 130, 137, 74, 56, 81, 13, 32, 4, 87, 71, 96, 92, 140, 1, 103, 144,
    68, 37, 134, 52, 106,
  ].map(x => x - 1)[year]
