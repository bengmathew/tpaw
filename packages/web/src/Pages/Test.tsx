import React, {useEffect, useState} from 'react'
import {extendTPAWParams} from '../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {runSimulationInWASM} from '../TPAWSimulator/Worker/RunSimulationInWASM'
import {useMarketData} from './App/WithMarketData'
import {Config} from './Config'

export const Test = React.memo(() => {
  const marketData = useMarketData()
  if (Config.client.production) throw new Error()

  useEffect(() => {
    // greet()
    // const x = foo()
    // console.dir(x)
    const importObject = {}
    // void WebAssembly.instantiateStreaming(
    //   fetch('simulator_bg.wasm'),
    //   importObject
    // ).then(results => {
    //   console.dir('oh')
    //   const x =  results.instance.exports.foo()
    //   console.dir(x)
    //   return 3

    // })
  }, [])
  const [rows, setRows] = useState<string[][]>([])

  useEffect(() => {
    void (async () => {
      const params = processTPAWParams(testParams, marketData)

      console.dir(
        params.returns.historicalAdjusted[indexIntoHistoricalReturns[0]]
      )
      const wasm = await runSimulationInWASM(
        params,
        {start: 0, end: 1},
        {
          truth: excel,
          indexIntoHistoricalReturns,
        }
      )

      // const resultsFromUsingExpectedReturns = runSPAWSimulation(params, {
      //   type: 'useExpectedReturns',
      // })

      // const result = runSPAWSimulation(params, {
      //   type: 'useHistoricalReturns',
      //   resultsFromUsingExpectedReturns,
      //   randomIndexesIntoHistoricalReturnsByYear,
      // })

      // const delta = result.byYearFromNow
      //   .map((x, i) => x.savingsPortfolio.end.balance)
      //   .map((x, i) => [
      //     `${i + params.people.person1.ages.current}`,
      //     `${x - excel[i]}`,
      //     `${formatCurrency(x)}`,
      //     `${x}`,
      //   ])
      // setRows(delta)
      // console.dir(result.byYearFromNow[56])
      // console.dir(prec.netPresentValue.savings.withCurrentYear[30])
      // console.dir(prec.netPresentValue.withdrawals.essential.withCurrentYear[30])
      // console.dir(
      //   prec.netPresentValue.withdrawals.discretionary.withCurrentYear[30]
      // )
      // console.dir(prec.netPresentValue.legacy.withCurrentYear[30])
      // console.dir(prec.cumulative1PlusGOver1PlusR[30])
      // console.dir(prec.netPresentValue.withdrawals.discretionary.withCurrentYear)
    })()
  }, [marketData])

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

const excelExpected = [
  77625, 106216.875, 135809.4656, 166437.7969, 198138.1198, 230947.954,
  264906.1324, 300052.847, 336429.6967, 374079.7361, 413047.5268, 453288.0726,
  494125.595, 535569.5818, 577629.68, 620315.6986, 663637.6118, 707605.5611,
  752229.8593, 797520.9924, 843489.6236, 890146.5957, 937502.9347, 985569.853,
  1034358.752, 1083881.227, 1134149.069, 1185174.269, 1236969.02, 1289545.724,
  1257788.592, 1225538.478, 1192787.563, 1159527.904, 1125751.432, 1091449.95,
  1056615.13, 1021238.512, 985311.5031, 948825.3724, 911771.2515, 874140.1316,
  835922.8607, 797110.1422, 757692.5323, 747960.4376, 737907.1133, 727525.6902,
  716809.1731, 705750.4385, 674142.2327, 662175.1691, 619316.7067, 606074.9796,
  592442.0093, 578409.6726, 563969.6996, 549113.6707, 533833.0146, 518119.0059,
  501962.7621, 485355.2413, 468287.2397, 450749.3887, 432732.1525, 414225.8252,
  395220.5281, 375706.2067, 355672.6283, 335109.3787, 314005.8595, 292351.285,
  270134.6795, 247344.8737, 223970.5022, 200000,
]
const excelSimulated = [
  92953.74286, 152577.9373, 217459.375, 253608.9506, 288246.7126, 233260.53,
  192314.8295, 221735.6608, 266990.5359, 198784.1206, 266908.9189, 334290.4443,
  388785.0988, 438282.4519, 448489.9419, 561179.5077, 572877.6501, 680030.6507,
  656964.3728, 726432.5146, 953657.2892, 824930.2865, 1108506.632, 1237168.815,
  1380477.14, 1594833.232, 1806001.788, 2040635.962, 2201290.769, 2116345.461,
  2417427.398, 2168606.395, 2015994.712, 2366832.209, 2452451.591, 2529563.247,
  1972133.479, 2236906.691, 2065862.673, 2155839.958, 2098779.577, 2160050.876,
  2253210.329, 2335110.417, 2205833.786, 2307539.837, 2329148.933, 2437928.931,
  2307341.912, 1846787.581, 2318596.05, 2405457.46, 2538800.156, 2419578.451,
  2170326.666, 2002626.026, 1809702.024, 1820418.377, 1879411.348, 1982032.057,
  1642515.017, 1484851.847, 1592363.745, 1486661.201, 1645226.889, 1733607.714,
  1855789.512, 1885606.818, 2006540.049, 2434884.281, 2215229.366, 1987915.381,
  1972423.2, 1793842.884, 1522433.054, 1248359.991,
]
const excel = excelSimulated

const testParams = extendTPAWParams({
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
    useTPAWPreset:false,
    tpawPreset:'riskLevel-2',
    customTPAWPreset:null,
    savedTPAWPreset:null,
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
const indexIntoHistoricalReturns = [
  30, 128, 147, 41, 25, 76, 76, 3, 18, 47, 93, 140, 18, 135, 92, 79, 137, 150,
  33, 97, 10, 71, 58, 24, 93, 85, 79, 35, 82, 11, 57, 46, 36, 58, 94, 27, 76,
  128, 131, 24, 25, 27, 18, 140, 129, 73, 136, 140, 33, 104, 112, 94, 45, 110,
  48, 96, 11, 122, 73, 56, 99, 6, 5, 86, 21, 133, 28, 123, 88, 112, 86, 120, 73,
  134, 87, 11,
].map(x => x - 1)
const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  indexIntoHistoricalReturns[year]
