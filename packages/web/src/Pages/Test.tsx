import { fGet, MAX_AGE_IN_MONTHS, ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import React, { useState } from 'react'
import { extendPlanParams } from '../TPAWSimulator/PlanParamsExt'
import { processPlanParams } from '../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { runSimulationInWASM } from '../TPAWSimulator/Worker/RunSimulationInWASM'
import { getTPAWRunInWorkerSingleton } from '../TPAWSimulator/Worker/UseTPAWWorker'
import { useMarketData } from './App/WithMarketData'
import { MarketData } from './Common/GetMarketData'
import { Config } from './Config'
1
export const Test = React.memo(() => {
  const marketData = useMarketData()
  if (Config.client.production) throw new Error()

  const [rows, setRows] = useState<string[][]>([])

  // useEffect(() => {
  //   void (async () => {
  // const params = processPlanParams(testParams, marketData)
  // console.dir('------------------------------')
  // console.dir('------------------------------')
  // console.dir('------------------------------')
  // const wasm = await runSimulationInWASM(
  //   params,
  //   { start: 0, end: 1 },
  //   {
  //     truth,
  //     indexIntoHistoricalReturns,
  //   },
  // )

  //     const adjustedBondReturns = params.returns.historicalMonthlyAdjusted.map(
  //       (x) => x.bonds,
  //     )
  //     console.dir(
  //       `Expected adjusted monthly: ${
  //         getStats(adjustedBondReturns).expectedValue
  //       }`,
  //     )
  //     console.dir(
  //       `Expected target monthly: ${annualToMonthlyReturnRate(
  //         params.returns.expectedAnnualReturns.bonds,
  //       )}`,
  //     )
  //     console.dir(
  //       `Expected adjusted annual: ${monthlyToAnnualReturnRate(
  //         getStats(adjustedBondReturns).expectedValue,
  //       )}`,
  //     )

  // const monthArrToYear = (year: number[]) =>
  //   year.map((x) => 1 + x).reduce((x, y) => x * y, 1) - 1

  //     const test = getStats(
  //       _.chunk(adjustedBondReturns, 12).map(monthArrToYear),
  //     ).expectedValue

  //     console.dir(`Expected adjusted annual2: ${test}`)

  //     const test2 = getStats(
  //       _.range(0, adjustedBondReturns.length - 12).map((i) =>
  //         monthArrToYear(adjustedBondReturns.slice(i, i + 12)),
  //       ),
  //     ).expectedValue
  //     console.dir(`Expected adjusted annual3: ${test2}`)
  //     console.dir(
  //       `Expected target annual: ${params.returns.expectedAnnualReturns.bonds}`,
  //     )

  //     console.dir(adjustedBondReturns.slice(0, 12).join('\n'))
  //     console.dir(
  //       adjustedBondReturns
  //         .slice(0, 12)
  //         .map((x) => 1 + x)
  //         .reduce((x, y) => x * y, 1) - 1,
  //     )

  //     // const stats = getStats(
  //     //   params.returns.historicalMonthlyAdjusted.map((x) => x.bonds),
  //     // )
  //     // const { historical, expected } = params.original.advanced.annualReturns
  //     // assert(historical.type === 'default')
  //     // assert(historical.adjust.type === 'toExpected')
  //     // const targetAnnualBondRate = params.returns.expectedAnnualReturns.bonds
  //     // const targetMonthlyBondRate =
  //     //   annualToMonthlyReturnRate(targetAnnualBondRate)
  //   })()
  // }, [marketData])

  return (
    <div className="">
      <button className="btn-dark btn-lg" onClick={handleExact(marketData)}>
        Test
      </button>
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

const handleExact = (marketData: MarketData) => async () => {
  const params = processPlanParams(testParams, marketData)
  console.dir('------------------------------')
  console.dir('------------------------------')
  console.dir('------------------------------')
  const url = new URL('https://dev.tpawplanner.com/plan')
  url.searchParams.set('params', JSON.stringify(params.original))
  console.dir(url.toString())
  const wasm = await runSimulationInWASM(
    params,
    { start: 0, end: 1 },
    {
      truth,
      indexIntoHistoricalReturns,
    },
  )
}

const handleStochastic = (marketData: MarketData) => async () => {
  const params = processPlanParams(testParams, marketData)
  const url = new URL('https://dev.tpawplanner.com/plan')
  url.searchParams.set('params', JSON.stringify(params.original))
  console.dir(url.toString())

  type Result = { 5: number[]; 50: number[]; 95: number[] }
  function mapResult<T>(
    x: Result,
    fn: (x: number[], percentile: 5 | 50 | 95) => T,
  ) {
    return { 5: fn(x[5], 5), 50: fn(x[50], 50), 95: fn(x[95], 95) }
  }

  let cumulative = null as null | Result
  const start = performance.now()
  const numIters = 1
  for (const i of _.range(0, numIters)) {
    const numRuns = 50000
    const percentiles = [5, 50, 95]
    const currFull = fGet(
      await getTPAWRunInWorkerSingleton().runSimulations(
        { canceled: false },
        numRuns,
        params,
        percentiles,
      ),
    )
    await getTPAWRunInWorkerSingleton().clearMemoizedRandom()
    const curr = {
      5: currFull.savingsPortfolio.start.balance.byPercentileByMonthsFromNow[0]
        .data,
      50: currFull.savingsPortfolio.start.balance.byPercentileByMonthsFromNow[1]
        .data,
      95: currFull.savingsPortfolio.start.balance.byPercentileByMonthsFromNow[2]
        .data,
    }
    if (!cumulative) {
      cumulative = curr
    } else {
      const next = mapResult(cumulative, (x, p) => _.zipWith(x, curr[p], _.add))

      const cumAvg = mapResult(cumulative, (x) => x.map((x) => x / i))
      const nextAvg = mapResult(next, (x) => x.map((x) => x / (i + 1)))

      const diff = (x: number[], y: number[]) =>
        _.zipWith(x, y, (x, y) => Math.abs(x - y))
      const maxDiff = fGet(
        _.max(
          diff(
            [...cumAvg[5], ...cumAvg[50], ...cumAvg[95]],
            [...nextAvg[5], ...nextAvg[50], ...nextAvg[95]],
          ),
        ),
      )
      console.dir(`Max Diff: ${maxDiff}`)
      cumulative = next
    }

    console.dir(`Completed: ${(i + 1) * numRuns}`)
  }

  const avgAnnual = mapResult(fGet(cumulative), (x) =>
    _.chunk(
      x.map((x) => x / numIters),
      12,
    ).map((x) => x[0]),
  )
  console.dir(mapResult(avgAnnual, _.last))
  console.dir(`Time: ${performance.now() - start}`)
}

const truth = _.flatten(
  [
    100000, 103567.04830000001, 105178.72483921982, 130106.38400172886,
    165153.31430013245, 191035.79918927536, 200342.16698580707,
    250375.57970040911, 294176.6869682576, 258148.8995921606, 292155.4866002399,
    282245.949083182, 297973.6067139881, 313461.457366743, 343596.1475930709,
    363123.7296502765, 362009.0461469589, 345182.7645451094, 372186.11553790985,
    396301.3948927011, 393810.7107711474, 435501.60453587666, 413922.7343309872,
    497300.6729645896, 548694.7078154874, 553729.655008101, 600269.8146528315,
    605655.9450589694, 629903.451075495, 553546.7298492829, 587699.7388499029,
    587782.5492693052, 548125.2245075032, 594672.6400947227, 603866.4310218884,
    627695.3258740185, 609997.0230512257, 636680.1698003262, 646135.9902422797,
    599332.2510441334, 664569.39156527, 697094.9128696055, 653995.4235340917,
    684899.2382583666, 702578.5868139802, 722209.3070050062, 659322.5806038221,
    560217.520279165, 500134.16433878354, 431892.3600177862, 456994.11414760974,
    573186.8418842127, 599160.9176883773, 621754.1632525818, 657526.0373345357,
    669766.3921296189, 730011.8469528913, 764104.1192672605, 782303.9393448607,
    831054.6237306581, 919958.76890975, 1029598.3041199942, 1219105.418428755,
    1250296.8547412287, 1285863.675529293, 1318092.3106634787,
  ].map((x) => _.times(12, () => x)),
)

const _convertByMonth = ({
  label,
  yearRange,
  value,
  nominal,
  id,
}: {
  label: string | null
  yearRange: {
    type: 'startAndEnd'
    start: { type: 'numericAge'; person: 'person1'; age: number }
    end: { type: 'numericAge'; person: 'person1'; age: number }
  }
  value: number
  nominal: boolean
  id: number
}) =>
  _.range(yearRange.start.age, yearRange.end.age + 1).map(
    (age, i): ValueForMonthRange => ({
      label,
      monthRange: {
        type: 'startAndNumMonths',
        start: { type: 'numericAge', person: 'person1', ageInMonths: age * 12 },
        numMonths: 1,
      },
      value,
      nominal,
      id: id * 100 + i,
    }),
  )

const testParams = extendPlanParams({
  v: 19,
  warnedAbout14to15Converstion: true,
  warnedAbout16to17Converstion: true,
  dialogPosition: 'done',
  people: {
    withPartner: false,
    person1: {
      ages: {
        type: 'notRetired',
        currentMonth: 35 * 12,
        retirementMonth: 65 * 12,
        maxMonth: 100 * 12,
      },
    },
  },
  wealth: {
    currentPortfolioBalance: 100000,
    futureSavings: [
      ..._convertByMonth({
        label: null,
        yearRange: {
          type: 'startAndEnd',
          start: { type: 'numericAge', person: 'person1', age: 37 },
          end: { type: 'numericAge', person: 'person1', age: 38 },
        },
        value: 12000,
        nominal: true,
        id: 1,
      }),
    ],
    retirementIncome: [
      ..._convertByMonth({
        label: null,
        yearRange: {
          type: 'startAndEnd',
          start: { type: 'numericAge', person: 'person1', age: 67 },
          end: { type: 'numericAge', person: 'person1', age: 68 },
        },
        value: 12000,
        nominal: true,
        id: 1,
      }),
    ],
  },
  adjustmentsToSpending: {
    tpawAndSPAW: {
      monthlySpendingCeiling: null,
      monthlySpendingFloor: null,
      legacy: {
        total: 1000000000,
        external: [],
      },
    },
    extraSpending: {
      essential: [
        ..._convertByMonth({
          label: null,
          yearRange: {
            type: 'startAndEnd',
            start: { type: 'numericAge', person: 'person1', age: 47 },
            end: { type: 'numericAge', person: 'person1', age: 48 },
          },
          value: 12000,
          nominal: true,
          id: 1,
        }),
        ..._convertByMonth({
          label: null,
          yearRange: {
            type: 'startAndEnd',
            start: { type: 'numericAge', person: 'person1', age: 47 },
            end: { type: 'numericAge', person: 'person1', age: 53 },
          },
          value: 12000,
          nominal: true,
          id: 2,
        }),
      ],
      discretionary: [],
    },
  },
  risk: {
    tpaw: {
      riskTolerance: {
        at20: 0,
        deltaAtMaxAge: 0,
        forLegacyAsDeltaFromAt20: 0,
      },
      timePreference: 0,
      additionalAnnualSpendingTilt: 0,
    },
    tpawAndSPAW: {
      lmp: 0,
    },
    spaw: { annualSpendingTilt: 0.0 },
    spawAndSWR: {
      allocation: {
        start: { stocks: 0.5 },
        intermediate: [],
        end: { stocks: 0.5 },
      },
    },
    swr: {
      withdrawal: { type: 'default' },
    },
  },

  advanced: {
    annualReturns: {
      expected: { type: 'manual', stocks: 0.04, bonds: 0.02 },
      // historical: {
      //   type: 'adjusted',
      //   adjustment: { type: 'toExpected' },
      //   correctForBlockSampling: true,
      // },
      historical: { type: 'unadjusted' },
    },
    annualInflation: { type: 'manual', value: 0.02 },
    sampling: 'monteCarlo',
    samplingBlockSizeForMonteCarlo: 12 * 1,
    strategy: 'TPAW',
  },
  dev: {
    alwaysShowAllMonths: false,
  },
})
const indexIntoHistoricalReturns = _.range(0, MAX_AGE_IN_MONTHS)
