import { File } from '@google-cloud/storage'
import {
  PLAN_PARAMS_CONSTANTS,
  MarketData,
  block,
  getNYZonedTime,
} from '@tpaw/common'
import { JSONGuard, number, object, string } from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { Clients } from '../../Clients.js'
import { Config } from '../../Config.js'
import { assert, fGet } from '../../Utils/Utils.js'
import { cliMisc } from './CLIMisc.js'

const lookBackInDays = 30
export type MarketData = Awaited<ReturnType<typeof _getMarketData>>['combined']

cliMisc
  .command('pushMarketData')
  .option('--printOnly')
  .action(
    async (options: { printOnly?: boolean }) =>
      await pushMarketData({ printOnly: options.printOnly }),
  )

const _printLatest = ({
  combined,
  raw,
}: Awaited<ReturnType<typeof _getMarketData>>) => {
  const formatDate = (x: number) =>
    getNYZonedTime(x).toLocaleString(DateTime.DATETIME_FULL)

  const x = fGet(combined[combined.length - 1])

  console.log(`--------------------`)
  console.log(`        Overall: ${formatDate(x.closingTime)}`)
  console.log('')
  console.log(`           SP500: ${formatDate(x.sp500.closingTime)}`)
  console.log(
    `       SP500-RAW: ${formatDate(fGet(_.last(raw.sp500)).closingTime)}`,
  )
  console.log('')
  console.log(`      bondRates: ${formatDate(x.bondRates.closingTime)}`)
  console.log(
    `  bondRates-RAW: ${formatDate(fGet(_.last(raw.bondRates)).closingTime)}`,
  )
  console.log('')
  console.log(`      inflation: ${formatDate(x.inflation.closingTime)}`)
  console.log(
    `  inflation-RAW: ${formatDate(fGet(_.last(raw.inflation)).closingTime)}`,
  )
  console.log('')
  console.log(
    `    stockMarket: ${formatDate(x.dailyStockMarketPerformance.closingTime)}`,
  )
  console.log(
    `stockMarket-RAW: ${formatDate(
      fGet(_.last(raw.dailyStockMarketPerformance)).closingTime,
    )}`,
  )
  console.log(`--------------------`)
  console.log(
    `vt: ${x.dailyStockMarketPerformance.percentageChangeFromLastClose.vt}`,
  )
  console.log(
    `bnd: ${x.dailyStockMarketPerformance.percentageChangeFromLastClose.bnd}`,
  )
  console.log('')
}

export const pushMarketData = async (opts: { printOnly?: boolean } = {}) => {
  const { combined, raw } = await _getMarketData()
  _printLatest({ combined, raw })
  if (opts.printOnly) return
  const bucket = Clients.gcs.bucket(Config.google.marketDataBucket)
  const [currentLatest] = await bucket.getFiles({ prefix: 'latest/' })
  await Promise.all(currentLatest.map((x) => x.delete()))
  const filename = getNYZonedTime.now().toFormat('yyyy-MM-dd_HH-mm-ss-ZZZZ')
  const files = [filename, `latest/${filename}`].map(
    (x) => new File(bucket, `${x}.json`),
  )
  for (const file of files) {
    await file.save(JSON.stringify(combined))
  }
  const latestFile = fGet(_.last(files))
  await latestFile.makePublic()
}

async function _getMarketData() {
  const [inflation, sp500, bondRates, dailyStockMarketPerformance] =
    await Promise.all([
      _getInflation(),
      _getSP500(),
      _getBondRates(),
      _getDailyStockMarketPerformance(),
    ])
  return {
    raw: { inflation, sp500, bondRates, dailyStockMarketPerformance },
    combined: MarketData.combineStreams(
      inflation,
      sp500,
      bondRates,
      dailyStockMarketPerformance,
    ),
  }
}

async function _getDailyStockMarketPerformance(): Promise<
  MarketData.DailyStockMarketPerformance[]
> {
  const [vtFull, bndFull] = await Promise.all([
    _getFromEOD('VT.US'),
    _getFromEOD('BND.US'),
  ])
  const minLength = Math.min(vtFull.length, bndFull.length)
  const vt = _.take(vtFull, minLength)
  const bnd = _.take(bndFull, minLength)
  const closingTimesAreIdentical = _.isEqualWith(
    vt.map((x) => x.closingTime),
    bnd.map((x) => x.closingTime),
  )
  assert(closingTimesAreIdentical)
  return _.zip(vt, bnd).map(([vt, bnd]) => ({
    closingTime: fGet(vt).closingTime,
    percentageChangeFromLastClose: {
      vt: fGet(vt).percentageChangeFromLastClose,
      bnd: fGet(bnd).percentageChangeFromLastClose,
    },
  }))
}

async function _getBondRates(): Promise<MarketData.BondRates[]> {
  const forYear = async (year: number) => {
    const url = new URL(
      `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/${year}/all`,
    )
    url.searchParams.set('type', 'daily_treasury_real_yield_curve')
    const response = await fetch(url, { headers: { cache: 'no-store' } })
    assert(response.ok)
    const text = await response.text()
    const rows = text.split('\n').map((x) => x.split(','))
    assert(
      _.isEqual(rows[0], [
        'Date',
        '"5 YR"',
        '"7 YR"',
        '"10 YR"',
        '"20 YR"',
        '"30 YR"',
      ]),
    )
    return rows.slice(1).map((cols) => ({
      closingTime: (() => {
        const [month, day, year] = fGet(cols[0]).split('/')
        return dateToMarketClosingTime(`${year}-${month}-${day}`)
      })(),
      fiveYear: fParsePercentString(fGet(cols[1])),
      sevenYear: fParsePercentString(fGet(cols[2])),
      tenYear: fParsePercentString(fGet(cols[3])),
      twentyYear: fParsePercentString(fGet(cols[4])),
      thirtyYear: fParsePercentString(fGet(cols[5])),
    }))
  }
  const result = _.sortBy(
    _.flatten(
      await Promise.all(
        _.range(
          DateTime.fromMillis(PLAN_PARAMS_CONSTANTS.minPlanParamTime).minus({
            days: lookBackInDays,
          }).year,
          // NY timezone since the data is coming from there.
          getNYZonedTime.now().year + 1,
        ).map(forYear),
      ),
    ),
    (x) => x.closingTime,
  )
  return result
}

// Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
async function _getInflation(): Promise<MarketData.Inflation[]> {
  const url = new URL('https://api.stlouisfed.org/fred/series/observations')
  url.searchParams.set('api_key', '369edd2764f90ce14eacc6aaef640fe3')
  url.searchParams.set('series_id', 'T10YIE')
  url.searchParams.set('file_type', 'json')
  url.searchParams.set(
    'observation_start',
    fGet(
      DateTime.fromMillis(PLAN_PARAMS_CONSTANTS.minPlanParamTime)
        .minus({ days: lookBackInDays })
        .toISODate(),
    ),
  )
  const response = await fetch(url, { headers: { cache: 'no-store' } })
  assert(response.ok)
  type ObType = { date: string; value: string }
  const guard: JSONGuard<ObType> = object(
    { date: string, value: string },
    'extraKeysOk',
  )
  const json = (await response.json()) as { observations: ObType[] }
  guard(json.observations[0]).force()
  return _.compact(
    json.observations.map((x) => {
      // This is because value for memorial day 2023-05-29 came back as "."
      // which fails parsing.
      const value = block(() => {
        try {
          return fParsePercentString(x.value)
        } catch (e) {
          return null
        }
      })
      return value
        ? {
            closingTime: dateToMarketClosingTime(x.date),
            value,
          }
        : null
    }),
  )
}

// // addedData is the date it was *added* to the array, not when it was true in the
// // world.
// const AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS = [
//   {
//     addedDate: DateTime.fromMillis(PLAN_PARAMS_CONSTANTS.minPlanParamTime)
//       .minus({ month: 30 })
//       .toMillis(),
//     tenYearDuration: {
//       start: { year: 2012, month: 10 },
//       end: { year: 2022, month: 9 },
//     },
//     value: 136.71,
//   },
//   {
//     // Monday, October 2, 2023 11:12:29.500 AM PDT
//     addedDate: 1696270349500,
//     tenYearDuration: {
//       start: { year: 2013, month: 4 },
//       end: { year: 2023, month: 3 },
//     },
//     value: 143.91,
//   },
//   {
//     // Sun Nov 05 2023 07:53:20 PST
//     addedDate: 1699199600576,
//     tenYearDuration: {
//       start: { year: 2013, month: 7 },
//       end: { year: 2023, month: 6 },
//     },
//     value: 146.14,
//   },
// ]

async function _getSP500(): Promise<MarketData.SP500[]> {
  return (await _getFromEOD('GSPC.INDX')).map((sp500) => {
    // const averageEarnings = fGet(
    //   _.findLast(
    //     AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
    //     (x) => sp500.closingTime > x.addedDate,
    //   ),
    // )
    // const oneOverCAPE = averageEarnings.value / sp500.close
    // const lnOnePlusOneOverCAPE = Math.log(1 + oneOverCAPE)
    // const regressionLog = {
    //   full: {
    //     fiveYear: 1.0152 * lnOnePlusOneOverCAPE - 0.0036,
    //     tenYear: 0.9279 * lnOnePlusOneOverCAPE + 0.00003,
    //     twentyYear: 0.6399 * lnOnePlusOneOverCAPE + 0.0179,
    //     thirtyYear: 0.2691 * lnOnePlusOneOverCAPE + 0.0434,
    //   },
    //   restricted: {
    //     fiveYear: 1.0192 * lnOnePlusOneOverCAPE + 0.0105,
    //     tenYear: 1.2114 * lnOnePlusOneOverCAPE - 0.0083,
    //     twentyYear: 0.9566 * lnOnePlusOneOverCAPE - 0.0016,
    //     thirtyYear: 0.3309 * lnOnePlusOneOverCAPE + 0.0396,
    //   },
    // }

    // const exp = (x: number) => Math.exp(x + 0.0141418028757969) - 1
    // const regression = {
    //   full: {
    //     fiveYear: exp(regressionLog.full.fiveYear),
    //     tenYear: exp(regressionLog.full.tenYear),
    //     twentyYear: exp(regressionLog.full.twentyYear),
    //     thirtyYear: exp(regressionLog.full.thirtyYear),
    //   },
    //   restricted: {
    //     fiveYear: exp(regressionLog.restricted.fiveYear),
    //     tenYear: exp(regressionLog.restricted.tenYear),
    //     twentyYear: exp(regressionLog.restricted.twentyYear),
    //     thirtyYear: exp(regressionLog.restricted.thirtyYear),
    //   },
    // }

    // const averageOfLowest4 = _.mean(
    //   _.sortBy(
    //     [
    //       oneOverCAPE,
    //       ..._.values(regression.full),
    //       ..._.values(regression.restricted),
    //     ],
    //     (x) => x,
    //   ).slice(0, 4),
    // )

    // const regressionAverage = _.mean([
    //   ..._.values(regression.full),
    //   ..._.values(regression.restricted),
    // ])

    return {
      closingTime: sp500.closingTime,
      value: sp500.close,
      // averageAnnualRealEarningsForSP500For10Years: averageEarnings,
      // value: 1 / oneOverCAPE,
      // oneOverCAPE,
      // regression,
      // regressionAverage,
      // suggested: averageOfLowest4,
    }
  })
}

// Solves the problem where _.round( parse('1.45') / 100, 3) becomes 0.014
// instead of 0.015.
const fParsePercentString = (x: string) => {
  const value = parseFloat(x)
  assert(!isNaN(value))
  const parts = x.trim().split('.')
  const precision = parts.length === 1 ? 0 : fGet(parts[1]).length
  return _.round(value / 100, precision + 2)
}

// Date format ISO 2020-03-22
const dateToMarketClosingTime = (date: string) =>
  DateTime.fromISO(`${date}`, {
    zone: fGet(getNYZonedTime.now().zoneName),
  })
    .set({ hour: 16 })
    .toMillis()

const _getFromEOD = async (name: string) => {
  const url = new URL(`https://eodhistoricaldata.com/api/eod/${name}`)
  url.searchParams.set('api_token', fGet(Config.eod.apiKey))
  url.searchParams.set('fmt', 'json')
  url.searchParams.set('period', 'd') // daily
  url.searchParams.set('order', 'a') // ascending
  url.searchParams.set(
    'from',
    fGet(
      DateTime.fromMillis(PLAN_PARAMS_CONSTANTS.minPlanParamTime)
        .minus({ days: lookBackInDays })
        .toISODate(),
    ),
  )

  const requestTime = DateTime.now()
  const response = await fetch(url, { headers: { cache: 'no-store' } })
  assert(response.ok)
  type EODData = {
    date: string // yyyy-mm-dd
    close: number
    adjusted_close: number
  }
  const guard: JSONGuard<EODData> = object(
    {
      date: string,
      close: number,
      adjusted_close: number,
    },
    'extraKeysOk',
  )
  const json = (await response.json()) as EODData[]
  guard(json[0]).force()
  const result = _.range(1, json.length).map((i) => {
    const curr = fGet(json[i])
    const prev = fGet(json[i - 1])
    return {
      closingTime: dateToMarketClosingTime(curr.date),
      percentageChangeFromLastClose:
        (curr.adjusted_close - prev.adjusted_close) / prev.adjusted_close,
      adjustedClose: curr.adjusted_close,
      close: curr.close,
    }
  })
  // For SP500 it can return intra-day data with, which looks like a closing
  // time in the future. Filter these out.
  return result.filter((x) => x.closingTime < requestTime.valueOf())
}

export const getMarketDataIndexForTime = (
  timestamp: number,
  marketData: MarketData,
) => {
  const index =
    _.sortedLastIndexBy<{ closingTime: number }>(
      marketData,
      { closingTime: timestamp },
      (x) => x.closingTime,
    ) - 1

  assert(index >= 0)
  return index
}

export const getMarketDataForTime = (
  timestamp: number,
  marketData: MarketData,
) => marketData[getMarketDataIndexForTime(timestamp, marketData)]
