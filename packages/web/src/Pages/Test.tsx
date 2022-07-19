import React, {useEffect, useState} from 'react'
import {extendTPAWParams} from '../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {runSimulationInWASM} from '../TPAWSimulator/Worker/RunSimulationInWASM'
import {Config} from './Config'

export const Test = React.memo(() => {
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
      const params = testParams

      const paramsExt = extendTPAWParams(params.original)
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

const excelExpected = [
  77625, 106216.875, 135809.4656, 166437.7969, 198138.1198, 230947.954,
  264906.1324, 300052.847, 336429.6967, 374079.7361, 413047.5268, 453379.1903,
  494313.706, 535858.8709, 578024.484, 620820.5113, 664257.0893, 708344.5278,
  753093.313, 798514.111, 844617.7706, 891415.3268, 938918.0044, 987137.2211,
  1036084.591, 1085771.929, 1136211.253, 1187414.788, 1239394.972, 1292164.456,
  1270913.674, 1248862.416, 1225992.662, 1202286.049, 1177723.855, 1152287.002,
  1125956.042, 1098711.155, 1070532.141, 1041398.414, 1011288.994, 980182.4998,
  948057.1446, 914890.7253, 880660.6171, 875643.7661, 869819.6809, 863167.4559,
  855665.7627, 847292.8436, 838026.5027, 817744.0991, 806521.538, 794335.253,
  781161.2075, 766974.8865, 731401.2877, 714758.7882, 697021.2779, 678162.1313,
  658154.1982, 636969.7942, 614580.6912, 590958.1081, 566072.7006, 539894.5514,
  512393.1602, 483537.433, 453295.6723, 421635.5659, 388524.1762, 353927.9296,
  317812.6049, 280143.3221, 240884.5312, 200000,
]
const excelSimulated = [
  80826.97796, 111450.7688, 128710.2261, 181618.7529, 196509.0923, 209725.4292,
  264998.5477, 306921.5479, 392044.3718, 449445.8166, 505333.5046, 585852.6074,
  665385.605, 738597.3514, 738343.8678, 794261.6523, 915964.5253, 830302.678,
  889945.0783, 997001.2347, 971995.6785, 746100.0964, 791001.7732, 781875.1888,
  839556.1262, 833243.5477, 978260.5649, 1132653.558, 1096430.762, 1223224.476,
  1072413.05, 948746.4815, 1071852.366, 1166508.863, 919156.7256, 849925.4989,
  944096.9056, 843158.6578, 793345.3152, 850360.7893, 880793.7249, 945021.1688,
  1012425.632, 943653.408, 947951.1577, 1011562.904, 1003526.163, 1073463.508,
  1124349.167, 1088928.887, 1023210.547, 1194674.99, 1233196.32, 1226249.942,
  1218875.675, 1107796.926, 1279375.009, 1405765.332, 1436231.845, 1549459.38,
  1272700.964, 1300740.939, 1346808.965, 1470046.418, 1557230.085, 1559326.442,
  1674631.674, 1589873.519, 1513458.722, 1873756.62, 2002270.715, 1878676.543,
  1772191.772, 2070246.799, 2314786.92, 2128909.063,
]
const excel = excelSimulated

const testParams = processTPAWParams(
  extendTPAWParams({
    v: 11,
    strategy: 'SWR',
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 25, retirement: 55, max: 100},
      },
    },
    returns: {
      expected: {type: 'suggested'},
      // historical: {type: 'default', adjust: {type: 'toExpected'}},
      historical: {
        type: 'default',
        // adjust: {
        //   type: 'to',
        //   stocks: 0.0345171123978209,
        //   bonds: 0.000467623297467001,
        // },
        adjust: {type: 'none'},
      },
    },
    inflation: {type: 'suggested'},
    targetAllocation: {
      regularPortfolio: {
        forTPAW: {stocks: 0.3},
        forSPAWAndSWR: {
          start: {stocks: 0.6},
          intermediate: [],
          end: {stocks: 0.6},
        },
      },
      legacyPortfolio: {stocks: 0.7},
    },
    swrWithdrawal: {type: 'asPercent', percent: 0.06},
    scheduledWithdrawalGrowthRate: 0.01,
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
      lmp: 0,
      essential: [
        // {
        //   id: 1,
        //   label: null,
        //   yearRange: {
        //     type: 'startAndNumYears',
        //     start: {type: 'numericAge', person: 'person1', age: 44},
        //     numYears: 1,
        //   },
        //   value: 30000,
        //   nominal: false,
        // },
        // {
        //   id: 2,
        //   label: null,
        //   yearRange: {
        //     type: 'startAndNumYears',
        //     start: {type: 'numericAge', person: 'person1', age: 75},
        //     numYears: 1,
        //   },
        //   value: 20000,
        //   nominal: false,
        // },
      ],
      discretionary: [
        // {
        //   id: 1,
        //   label: null,
        //   yearRange: {
        //     type: 'startAndNumYears',
        //     start: {type: 'numericAge', person: 'person1', age: 77},
        //     numYears: 1,
        //   },
        //   value: 30000,
        //   nominal: false,
        // },
      ],
    },
    spendingCeiling: null,
    spendingFloor: null,
    legacy: {
      total: 200000,
      external: [],
    },
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  })
)
const indexIntoHistoricalReturns = [
  102, 95, 11, 105, 124, 20, 144, 78, 28, 102, 40, 140, 142, 114, 36, 69, 80,
  37, 69, 24, 124, 47, 98, 87, 69, 109, 119, 144, 64, 81, 131, 96, 75, 30, 71,
  86, 38, 59, 83, 28, 5, 85, 133, 86, 142, 73, 110, 80, 18, 108, 120, 115, 26,
  83, 135, 46, 21, 55, 141, 143, 138, 3, 26, 126, 18, 89, 140, 42, 120, 125,
  123, 49, 32, 52, 139, 117,
].map(x => x - 1)
const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  indexIntoHistoricalReturns[year]
