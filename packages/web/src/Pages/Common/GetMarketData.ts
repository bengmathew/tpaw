import {
  MIN_PLAN_PARAM_TIME,
  NonPlanParams,
  PlanParams,
  noCase,
} from '@tpaw/common'
import { JSONGuard, number, object, string } from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { planParamsProcessExpectedAnnualReturns } from '../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessAnnualReturns'
import { assert, fGet } from '../../Utils/Utils'
import { Config } from '../Config'

const lookBackInDays = 30
export type MarketData = Awaited<ReturnType<typeof getMarketData>>

export async function getMarketData() {
  let [inflation, CAPE, bondRates, dailyStockMarketPerformance] =
    await Promise.all([
      getInflation(),
      getCAPE(),
      getBondRates(),
      getDailyStockMarketPerformance(),
    ])
  // const smallestLastClosingTimes = Math.min(
  //   ...[inflation, CAPE, bondRates, dailyStockMarketPerformance].map(
  //     (x) => x[x.length - 1].closingTime,
  //   ),
  // )
  const largestFirstClosingTimes = Math.max(
    ...[inflation, CAPE, bondRates, dailyStockMarketPerformance].map(
      (x) => x[0].closingTime,
    ),
  )
  const formatToNY = (x: number) =>
    DateTime.fromMillis(x, {
      zone: NYTimeZone,
    }).toLocaleString(DateTime.DATETIME_FULL)
  console.log('-------------------------------------')
  console.log('Market Data')
  console.log('-------------------------------------')
  console.log(`From: ${formatToNY(largestFirstClosingTimes)}`)
  console.log('To:')
  console.log(` inflation: ${formatToNY(fGet(_.last(inflation)).closingTime)}`)
  console.log(`      CAPE: ${formatToNY(fGet(_.last(CAPE)).closingTime)}`)
  console.log(` bondRates: ${formatToNY(fGet(_.last(bondRates)).closingTime)}`)
  console.log(
    `VT and BND: ${formatToNY(
      fGet(_.last(dailyStockMarketPerformance)).closingTime,
    )}`,
  )
  console.log('-------------------------------------')
  const filter = <T extends { closingTime: number }>(x: T[]) => {
    return x.filter(
      (x) =>
        // x.closingTime <= smallestLastClosingTimes &&
        x.closingTime >= largestFirstClosingTimes,
    )
  }
  inflation = filter(inflation)
  CAPE = filter(CAPE)
  bondRates = filter(bondRates)
  dailyStockMarketPerformance = filter(dailyStockMarketPerformance)

  const lengths = [inflation, CAPE, bondRates, dailyStockMarketPerformance].map(
    (x) => x.length,
  )
  const minLength = Math.min(...lengths)
  const maxLength = Math.max(...lengths)
  assert(maxLength <= minLength + 1)
  assert(
    _.isEqual(
      inflation.map((x) => x.closingTime).slice(0, minLength),
      CAPE.map((x) => x.closingTime).slice(0, minLength),
    ),
  )
  assert(
    _.isEqual(
      inflation.map((x) => x.closingTime).slice(0, minLength),
      bondRates.map((x) => x.closingTime).slice(0, minLength),
    ),
  )
  assert(
    _.isEqual(
      inflation.map((x) => x.closingTime).slice(0, minLength),
      dailyStockMarketPerformance.map((x) => x.closingTime).slice(0, minLength),
    ),
  )

  const latest = {
    inflation: fGet(_.last(inflation)),
    CAPE: fGet(_.last(CAPE)),
    bondRates: fGet(_.last(bondRates)),
  }
  const result = {
    CAPE,
    inflation,
    bondRates,
    dailyStockMarketPerformance,
    latest,
  }
  return result
}

export async function getDailyStockMarketPerformance() {
  const [vt, bnd] = await Promise.all([
    _getFromEOD('VT.US'),
    _getFromEOD('BND.US'),
  ])
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

async function getBondRates() {
  const forYear = async (year: number) => {
    const url = new URL(
      `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/${year}/all`,
    )
    url.searchParams.set('type', 'daily_treasury_real_yield_curve')
    const response = await fetch(url)
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
        const [month, day, year] = cols[0].split('/')
        return dateToMarketClosingTime(`${year}-${month}-${day}`)
      })(),
      fiveYear: fParsePercentString(cols[1]),
      sevenYear: fParsePercentString(cols[2]),
      tenYear: fParsePercentString(cols[3]),
      twentyYear: fParsePercentString(cols[4]),
      thirtyYear: fParsePercentString(cols[5]),
    }))
  }
  const result = _.sortBy(
    _.flatten(
      await Promise.all(
        _.range(
          DateTime.fromMillis(MIN_PLAN_PARAM_TIME).minus({
            days: lookBackInDays,
          }).year,
          // NY timezone since the data is coming from there.
          DateTime.local().setZone(NYTimeZone).year + 1,
        ).map(forYear),
      ),
    ),
    (x) => x.closingTime,
  )
  return result
}

// Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
async function getInflation() {
  const url = new URL('https://api.stlouisfed.org/fred/series/observations')
  url.searchParams.set('api_key', '369edd2764f90ce14eacc6aaef640fe3')
  url.searchParams.set('series_id', 'T10YIE')
  url.searchParams.set('file_type', 'json')
  url.searchParams.set(
    'observation_start',
    DateTime.fromMillis(MIN_PLAN_PARAM_TIME)
      .minus({ days: lookBackInDays })
      .toISODate(),
  )
  const response = await fetch(url)
  assert(response.ok)
  type ObType = { date: string; value: string }
  const guard: JSONGuard<ObType> = object(
    { date: string, value: string },
    'extraKeysOk',
  )
  const json = (await response.json()) as { observations: ObType[] }
  guard(json.observations[0]).force()
  const result = json.observations.map((x) => ({
    closingTime: dateToMarketClosingTime(x.date),
    value: fParsePercentString(x.value),
  }))
  return result
}

// addedData is the date it was *added* to the array, not when it was true in the
// world.
const AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS = [
  {
    addedDate: DateTime.fromMillis(MIN_PLAN_PARAM_TIME)
      .minus({ month: 30 })
      .valueOf(),
    value: 136.71,
  },
]

async function getCAPE() {
  const sp500 = await _getFromEOD('GSPC.INDX')
  return sp500.map(({ closingTime, close }) => {
    const averageEarnings = fGet(
      _.findLast(
        AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
        (x) => closingTime > x.addedDate,
      ),
    )
    const oneOverCAPE = averageEarnings.value / close
    const lnOnePlusOneOverCAPE = Math.log(1 + oneOverCAPE)
    const regressionLog = {
      full: {
        fiveYear: 1.0152 * lnOnePlusOneOverCAPE - 0.0036,
        tenYear: 0.9279 * lnOnePlusOneOverCAPE + 0.00003,
        twentyYear: 0.6399 * lnOnePlusOneOverCAPE + 0.0179,
        thirtyYear: 0.2691 * lnOnePlusOneOverCAPE + 0.0434,
      },
      restricted: {
        fiveYear: 1.0192 * lnOnePlusOneOverCAPE + 0.0105,
        tenYear: 1.2114 * lnOnePlusOneOverCAPE - 0.0083,
        twentyYear: 0.9566 * lnOnePlusOneOverCAPE - 0.0016,
        thirtyYear: 0.3309 * lnOnePlusOneOverCAPE + 0.0396,
      },
    }

    const exp = (x: number) => Math.exp(x + 0.0141418028757969) - 1
    const regression = {
      full: {
        fiveYear: exp(regressionLog.full.fiveYear),
        tenYear: exp(regressionLog.full.tenYear),
        twentyYear: exp(regressionLog.full.twentyYear),
        thirtyYear: exp(regressionLog.full.thirtyYear),
      },
      restricted: {
        fiveYear: exp(regressionLog.restricted.fiveYear),
        tenYear: exp(regressionLog.restricted.tenYear),
        twentyYear: exp(regressionLog.restricted.twentyYear),
        thirtyYear: exp(regressionLog.restricted.thirtyYear),
      },
    }

    const averageOfLowest4 = _.mean(
      _.sortBy(
        [
          oneOverCAPE,
          ..._.values(regression.full),
          ..._.values(regression.restricted),
        ],
        (x) => x,
      ).slice(0, 4),
    )

    const regressionAverage = _.mean([
      ..._.values(regression.full),
      ..._.values(regression.restricted),
    ])

    return {
      closingTime,
      value: 1 / oneOverCAPE,
      oneOverCAPE,
      regression,
      regressionAverage,
      suggested: averageOfLowest4,
    }
  })
}

// Solves the problem where _.round( parse('1.45') / 100, 3) becomes 0.014
// instead of 0.015.
const fParsePercentString = (x: string) => {
  const value = parseFloat(x)
  assert(!isNaN(value))
  const parts = x.trim().split('.')
  const precision = parts.length === 1 ? 0 : parts[1].length
  return _.round(value / 100, precision + 2)
}

// Date format ISO 2020-03-22
const dateToMarketClosingTime = (date: string) =>
  DateTime.fromISO(date, { zone: NYTimeZone }).set({ hour: 16 }).valueOf()

const _getFromEOD = async (name: string) => {
  const url = new URL(`https://eodhistoricaldata.com/api/eod/${name}`)
  url.searchParams.set('api_token', fGet(Config.server.eod.apiKey))
  url.searchParams.set('fmt', 'json')
  url.searchParams.set('period', 'd') // daily
  url.searchParams.set('order', 'a') // ascendign
  url.searchParams.set(
    'from',
    DateTime.fromMillis(MIN_PLAN_PARAM_TIME)
      .minus({ days: lookBackInDays })
      .toISODate(),
  )

  const requestTime = DateTime.now()
  const response = await fetch(url)
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
    const curr = json[i]
    const prev = json[i - 1]
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

const NYTimeZone = 'America/New_York'

export const synthesizeMarketDataForTesting = (
  marketData: MarketData,
  futureTime: number,
  strategy: Extract<
    NonPlanParams['dev']['currentTimeFastForward'],
    { shouldFastForward: true }
  >['marketDataExtensionStrategy'],

  expectedAnnualReturnsIn: PlanParams['advanced']['annualReturns'],
): MarketData => {
  const start = performance.now()
  const helper = <T extends { closingTime: number }>(
    original: T[],
    getValue: (
      daysSinceLastMarketClose: number,
      index: number,
      original: T[],
    ) => Omit<T, 'closingTime'> = (_, i, original) =>
      original[i % original.length],
  ) => {
    let marketCloses = _synthesizeMarketCloseTimes(
      original.map((x) => x.closingTime),
      futureTime,
    )
    return marketCloses.map((closingTime, i) => {
      const result = (x: Omit<T, 'closingTime'>) => ({ ...x, closingTime })
      const lastMarketClose =
        i === 0
          ? _marketCloseDelta(closingTime, -1).valueOf()
          : marketCloses[i - 1]
      const numDaysSinceLastMarketClose = Math.round(
        fGet(
          DateTime.fromMillis(marketCloses[i])
            .diff(DateTime.fromMillis(lastMarketClose), 'days')
            .toObject().days,
        ),
      )
      return result(getValue(numDaysSinceLastMarketClose, i, original))
    })
  }

  const inflation = helper(marketData.inflation)
  const CAPE = helper(marketData.CAPE)
  const bondRates = helper(marketData.bondRates)
  const latest = {
    inflation: fGet(_.last(inflation)),
    CAPE: fGet(_.last(CAPE)),
    bondRates: fGet(_.last(bondRates)),
  }

  const latestExpectedAnnualReturns = planParamsProcessExpectedAnnualReturns(
    expectedAnnualReturnsIn,
    latest,
  )

  const dailyStockMarketPerformance = helper(
    marketData.dailyStockMarketPerformance,
    strategy.dailyStockMarketPerformance === 'latestExpected'
      ? (daysSinceLastMarketClose) => {
          const fromAnnual = (annual: number) =>
            Math.pow(1 + annual, daysSinceLastMarketClose / 365) - 1
          return {
            percentageChangeFromLastClose: {
              vt: fromAnnual(latestExpectedAnnualReturns.stocks),
              bnd: fromAnnual(latestExpectedAnnualReturns.bonds),
            },
          }
        }
      : strategy.dailyStockMarketPerformance === 'repeatGrowShrinkZero'
      ? (_, i) => {
          const _fromGrow = (grow: number) =>
            [grow, 1 / (1 + grow) - 1, 0][i % 3]
          return {
            percentageChangeFromLastClose: {
              vt: _fromGrow(0.05),
              bnd: _fromGrow(0.05),
            },
          }
        }
      : strategy.dailyStockMarketPerformance === 'roundRobinPastValues'
      ? undefined
      : noCase(strategy.dailyStockMarketPerformance),
  )

  const result = {
    CAPE,
    inflation,
    bondRates,
    dailyStockMarketPerformance,
    latest,
  }
  return result
}

let _futureMarketCloses = [] as number[]
const _synthesizeMarketCloseTimes = (original: number[], end: number) => {
  const lastOriginal = fGet(_.last(original))
  let next = _marketCloseDelta(lastOriginal, 1).valueOf()
  if (_futureMarketCloses.length === 0) _futureMarketCloses.push(next)
  while (next < _futureMarketCloses[0]) {
    _futureMarketCloses.push(next)
    next = _marketCloseDelta(next, 1).valueOf()
  }
  const nextFromEnd = _marketCloseDelta(
    DateTime.fromMillis(end, { zone: NYTimeZone }).set({
      hour: 16,
      minute: 0,
      millisecond: 0,
    }),
    1,
  ).valueOf()
  while (fGet(_.last(_futureMarketCloses)) < nextFromEnd) {
    _futureMarketCloses.push(
      _marketCloseDelta(fGet(_.last(_futureMarketCloses)), 1).valueOf(),
    )
  }
  const futureIndex = _futureMarketCloses.findIndex((x) => x > lastOriginal)
  return [
    ...original,
    ...(futureIndex === -1 ? [] : _futureMarketCloses.slice(futureIndex)),
  ].filter((x) => x < end)
}

const _marketCloseDelta = (
  currClosingTime: number | DateTime,
  delta: 1 | -1,
) => {
  let next = DateTime.fromMillis(currClosingTime.valueOf(), {
    zone: NYTimeZone,
  })
  do {
    next = next.plus({ day: delta })
  } while (next.weekdayShort === 'Sun' || next.weekdayShort === 'Sat')
  return next.set({ hour: 16, minute: 0, second: 0, millisecond: 0 })
}
