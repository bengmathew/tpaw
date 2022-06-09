import React, {useEffect, useState} from 'react'
import {historicalReturns} from '../TPAWSimulator/HistoricalReturns'
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

      const prec = params.preCalculations.forSPAW
      const paramsExt = extendTPAWParams(params.original)
      console.dir(historicalReturns)
      console.dir(params.returns.historicalAdjusted)
      const wasm = (
        await runSimulationInWASM(params, 1, {
          truth: excel,
          indexIntoHistoricalReturns,
        })
      ).result

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
  68724.32659, 105682.0022, 160005.1932, 200281.7845, 278918.9695, 326463.9137,
  414238.3968, 579581.5432, 674932.5005, 782909.2687, 945286.7643, 1008131.011,
  939678.4308, 1020347.811, 976670.4534, 928311.3948, 998828.1304, 1160115.328,
  1412466.932, 1476493.733, 1687382.179, 1577910.38, 1651912.555, 2135427.424,
  2078536.241, 2151957.821, 2386837.044, 2311433.747, 1958239.626, 2122864.198,
  2103483.881, 1981400.844, 2248939.078, 2037583.765, 2013781.796, 2294081.567,
  2312850.817, 2207038.215, 2712734.261, 2633469.884, 2553077.342, 2564161.146,
  2214668.225, 1922032.373, 1898432.507, 1922079.482, 1869128.335, 1943408.246,
  1758733.976, 1783210.686, 1765693.032, 1818383.535, 2026309.328, 2037823.993,
  2019349.727, 2016983.111, 2076256.063, 1874269.41, 1804255.006, 1431024.57,
  1325942.849, 1247789.444, 1343013.889, 1382954.837, 1348927.325, 1301731.354,
  1090871.744, 1021647.358, 999231.689, 895481.6112, 735210.8837, 650657.9804,
  718623.795, 764171.1646, 683857.1311, 570585.245,
]
const excel = excelSimulated

const testParams = processTPAWParams({
  v: 7,
  strategy: 'TPAW',
  people: {
    withPartner: false,
    person1: {
      displayName: null,
      ages: {type: 'notRetired', current: 25, retirement: 55, max: 100},
    },
  },
  returns: {
    expected: {stocks: 0.035, bonds: 0.01},
    // historical: {type: 'default', adjust: {type: 'toExpected'}},
    historical: {
      type: 'default',
      adjust: {type: 'by', stocks: 0.048, bonds: 0.03},
    },
  },
  inflation: 0.02,
  targetAllocation: {
    regularPortfolio: {
      forTPAW: {stocks: 0.3},
      forSPAW: {
        start: {stocks: 0.6},
        intermediate: [],
        end: {stocks: 0.6},
      },
    },
    legacyPortfolio: {stocks: 0.7},
  },
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
      {
        id: 2,
        label: null,
        yearRange: {
          type: 'startAndNumYears',
          start: {type: 'numericAge', person: 'person1', age: 76},
          numYears: 1,
        },
        value: 10000,
        nominal: false,
      },
    ],
    discretionary: [
      {
        id: 1,
        label: null,
        yearRange: {
          type: 'startAndNumYears',
          start: {type: 'numericAge', person: 'person1', age: 81},
          numYears: 1,
        },
        value: 20000,
        nominal: false,
      },
    ],
  },
  spendingCeiling: null,
  spendingFloor: null,
  legacy: {
    total: 200000,
    external: [],
  },
  display: {
    alwaysShowAllYears: false,
  },
})
const indexIntoHistoricalReturns = [
  92, 91, 85, 144, 52, 118, 143, 88, 68, 55, 52, 16, 50, 31, 44, 32, 18, 149, 8,
  3, 52, 6, 113, 112, 43, 118, 28, 60, 103, 56, 113, 92, 115, 145, 16, 58, 14,
  137, 112, 136, 2, 14, 46, 48, 90, 94, 9, 79, 148, 140, 118, 79, 57, 73, 3, 23,
  5, 39, 78, 67, 98, 83, 75, 5, 80, 142, 117, 78, 126, 110, 131, 135, 58, 116,
  101, 135,
].map(x => x - 1)
const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  indexIntoHistoricalReturns[year]
