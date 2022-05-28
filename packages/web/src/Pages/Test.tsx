import React, {useEffect, useState} from 'react'
import {extendTPAWParams} from '../TPAWSimulator/TPAWParamsExt'
import {processTPAWParams} from '../TPAWSimulator/TPAWParamsProcessed'
import {runSPAWSimulation} from '../TPAWSimulator/Worker/RunSPAWSimulation'
import {formatCurrency} from '../Utils/FormatCurrency'
import {Config} from './Config'

export const Test = React.memo(() => {
  if (Config.client.production) throw new Error()

  const [rows, setRows] = useState<string[][]>([])

  useEffect(() => {
    const params = testParams

    const prec = params.preCalculations.forSPAW
    const paramsExt = extendTPAWParams(params.original)

    const resultsFromUsingExpectedReturns = runSPAWSimulation(
      params,
      paramsExt,
      {
        type: 'useExpectedReturns',
      }
    )

    const result = runSPAWSimulation(params, paramsExt, {
      type: 'useHistoricalReturns',
      resultsFromUsingExpectedReturns,
      randomIndexesIntoHistoricalReturnsByYear,
    })

    const delta = result.byYearFromNow
      .map((x, i) => x.savingsPortfolio.end.balance)
      .map((x, i) => [
        `${i + params.people.person1.ages.current}`,
        `${x - excel[i]}`,
        `${formatCurrency(x)}`,
        `${x}`,
      ])
    setRows(delta)
    console.dir(result.byYearFromNow[56])
    console.dir(prec.netPresentValue.savings.withCurrentYear[30])
    console.dir(prec.netPresentValue.withdrawals.essential.withCurrentYear[30])
    console.dir(
      prec.netPresentValue.withdrawals.discretionary.withCurrentYear[30]
    )
    console.dir(prec.netPresentValue.legacy.withCurrentYear[30])
    console.dir(prec.cumulative1PlusGOver1PlusR[30])
    console.dir(prec.netPresentValue.withdrawals.discretionary.withCurrentYear)
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
  79932.81452, 109786.50362, 151083.14282, 182999.86374, 254436.66079,
  287089.15811, 264753.62134, 313417.46376, 290242.63902, 296457.81508,
  322005.14813, 363055.94322, 449284.25017, 453628.78432, 492249.07174,
  540627.5658, 525150.53761, 565216.65494, 633168.69608, 644941.28896,
  538153.84735, 590038.30534, 652965.59417, 716753.35105, 674484.42086,
  731099.54262, 726328.10785, 680518.80749, 714136.79734, 792399.23765,
  816995.59796, 773042.08999, 794794.62078, 749692.14887, 708583.02223,
  693702.98424, 601503.32117, 629470.06741, 624482.816, 584184.66816,
  505828.91573, 489838.94401, 456156.51348, 436340.03455, 414109.70815,
  356417.05778, 418717.38705, 410662.99794, 412600.34624, 483136.69768,
  453735.76029, 386230.26406, 339914.06321, 330871.57418, 291778.89355,
  330797.10973, 271687.29244, 272395.50196, 244478.55945, 173964.65771,
  156318.33971, 158093.38202, 154265.15769, 170628.00249, 183055.40027,
  169884.25679, 169316.21456, 164006.03266, 151612.7949, 132114.56017,
  145988.86312, 118718.6334, 88718.04282, 72872.73962, 59038.7291, 40834.66815,
]

const testParams = processTPAWParams({
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
      forTPAW: {stocks: 0.3},
      forSPAW: {
        start: {stocks: 0.6},
        intermediate: [],
        end: {stocks: 0.6},
      },
    },
    legacyPortfolio: {stocks: 0.7},
  },
  scheduledWithdrawalGrowthRate: 0.0,
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
          start: {type: 'numericAge', person: 'person1', age: 75},
          numYears: 1,
        },
        value: 20000,
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
        value: 30000,
        nominal: false,
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
const randomIndexesIntoHistoricalReturnsByYear = (year: number) =>
  [
    19, 24, 30, 101, 125, 114, 29, 126, 111, 43, 23, 24, 88, 92, 26, 142, 145,
    114, 73, 134, 103, 113, 74, 94, 20, 142, 39, 11, 122, 80, 68, 23, 55, 23,
    97, 82, 36, 45, 91, 14, 43, 74, 90, 74, 27, 131, 125, 40, 9, 125, 114, 6,
    109, 136, 145, 51, 43, 19, 92, 76, 39, 144, 146, 51, 38, 72, 35, 150, 16,
    135, 65, 39, 60, 136, 19, 141,
  ].map(x => x - 1)[year]
