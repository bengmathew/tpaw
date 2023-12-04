import {
  MAX_AGE_IN_MONTHS,
  calendarMonthFromTime,
  currentPlanParamsVersion,
  fGet,
} from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useState } from 'react'
import { extendPlanParams } from '../TPAWSimulator/ExtentPlanParams'
import { Config } from './Config'
import { useMarketData } from './PlanRoot/PlanRootHelpers/WithMarketData'

export const Test = React.memo(() => {
  const marketData = useMarketData()
  const [currentTime] = useState(DateTime.local)
  if (Config.client.isProduction) throw new Error()

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
      <button
        className="btn-dark btn-lg"
        // onClick={handleExact(marketData, currentTime)}
      >
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

// const handleExact =
//   (marketData: MarketData, currentTime: DateTime) => async () => {
//     const testParams = getTestParams(currentTime)
//     const params = processPlanParams(
//       testParams,
//       testParams.planParams.wealth.portfolioBalance.amount,
//       marketData.latest,
//     )
//     const wasm = runSimulationInWASM(
//       params,
//       { start: 0, end: 1 },
//       await getWASM(),
//       {
//         truth,
//         indexIntoHistoricalReturns,
//       },
//     )
//   }

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

// const _convertByMonth = ({
//   label,
//   yearRange,
//   value,
//   nominal,
//   id,
// }: {
//   label: string | null
//   yearRange: {
//     type: 'startAndEnd'
//     start: { type: 'numericAge'; person: 'person1'; age: number }
//     end: { type: 'numericAge'; person: 'person1'; age: number }
//   }
//   value: number
//   nominal: boolean
//   id: number
// }) =>
//   _.range(yearRange.start.age, yearRange.end.age + 1).map(
//     (age, i): ValueForMonthRange => ({
//       label,
//       monthRange: {
//         type: 'startAndNumMonths',
//         start: {
//           type: 'numericAge',
//           person: 'person1',
//           age: { inMonths: age * 12 },
//         },
//         numMonths: 1,
//       },
//       value,
//       nominal,
//       id: id * 100 + i,
//     }),
//   )

const getTestParams = (currentTime: DateTime) =>
  extendPlanParams(
    {
      v: currentPlanParamsVersion,
      results: null,
      timestamp: currentTime.valueOf(),
      dialogPositionNominal: 'done',
      people: {
        withPartner: false,
        person1: {
          ages: {
            type: 'retirementDateSpecified',
            monthOfBirth: calendarMonthFromTime(
              currentTime.minus({ month: 35 * 12 }),
            ),
            retirementAge: { inMonths: 65 * 12 },
            maxAge: { inMonths: 100 * 12 },
          },
        },
      },
      wealth: {
        portfolioBalance: {
          updatedHere: true,
          amount: 100000,
        },
        futureSavings: {},
        incomeDuringRetirement: {},
      },
      adjustmentsToSpending: {
        tpawAndSPAW: {
          monthlySpendingCeiling: null,
          monthlySpendingFloor: null,
          legacy: {
            total: 1000000000,
            external: {},
          },
        },
        extraSpending: {
          essential: {},
          discretionary: {},
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
            start: {
              month: calendarMonthFromTime(currentTime),
              stocks: 0.5,
            },
            intermediate: {},
            end: { stocks: 0.5 },
          },
        },
        swr: {
          withdrawal: { type: 'default' },
        },
      },

      advanced: {
        annualInflation: { type: 'suggested' },
        expectedAnnualReturnForPlanning: {
          type: 'manual',
          stocks: 0.04,
          bonds: 0.02,
        },
        historicalReturnsAdjustment: {
          stocks: {
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
            volatilityScale: 1,
          },
          bonds: {
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
            enableVolatility: true,
          },
        },
        sampling: {
          type: 'monteCarlo',
          blockSizeForMonteCarloSampling: 12 * 5,
        },
        strategy: 'TPAW',
      },
    },
    currentTime.toMillis(),
    fGet(currentTime.zoneName),
  )
const indexIntoHistoricalReturns = _.range(0, MAX_AGE_IN_MONTHS)
