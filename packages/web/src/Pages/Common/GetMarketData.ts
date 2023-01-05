import { assert } from '../../Utils/Utils'

import { Element } from 'domhandler'
import { findAll, textContent } from 'domutils'
import * as htmlparser2 from 'htmlparser2'
import _ from 'lodash'

export type MarketData = Awaited<ReturnType<typeof getMarketData>>

export async function getMarketData() {
  const inflation = await getInflation()
  const CAPE = await getCAPE()
  const bondRates = await getBondRates()
  return { CAPE, inflation, bondRates }
}

async function getBondRates() {
  const year = new Date().getFullYear()
  const rowsForYear = async (year: number) => {
    const response = await fetch(
      `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/${year}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value=${year}&page&_format=csv`,
    )
    assert(response.ok)
    const text = await response.text()
    const rows = text.split('\n')
    return rows
  }
  let rows = await rowsForYear(year)
  if (rows.length < 2) rows = await rowsForYear(year - 1)
  return _parseBondRow(rows[1])
}

function _parseBondRow(row: string) {
  const cols = row.split(',')
  return {
    date: fParseDate(cols[0]),
    fiveYear: fParseFloat(cols[1]) / 100,
    sevenYear: fParseFloat(cols[2]) / 100,
    tenYear: fParseFloat(cols[3]) / 100,
    twentyYear: fParseFloat(cols[4]) / 100,
    thirtyYear: fParseFloat(cols[5]) / 100,
  }
}

async function getInflation() {
  const date = new Date()
  const oneYearAgoStr = `${date.getFullYear() - 1}-${
    date.getMonth() + 1
  }-${date.getDate()}`
  const response = await fetch(
    `https://fred.stlouisfed.org/graph/fredgraph.csv?mode=fred&id=T10YIE&cosd=${oneYearAgoStr}`,
  )
  assert(response.ok)
  const text = await response.text()
  const rows = text.split('\n')
  return _parseInflationRow(rows[rows.length - 2])
}

function _parseInflationRow(row: string) {
  const cols = row.split(',')
  return {
    date: fParseDate(cols[0]),
    value: fParseFloat(cols[1]) / 100,
  }
}

async function getCAPE() {
  const response = await fetch(
    'https://www.multpl.com/shiller-pe/table/by-month',
  )
  assert(response.ok)
  const htmlString = await response.text()
  const dom = htmlparser2.parseDocument(htmlString)
  const rows = findAll((e) => e.tagName === 'tr', dom.childNodes)
  const { date, value } = _parseCAPERow(rows[1])
  const oneOverCAPE = 1 / value
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
    date,
    value: value,
    oneOverCAPE: oneOverCAPE,
    regression,
    regressionAverage,
    suggested: averageOfLowest4,
  }
}

const _parseCAPERow = (row: Element) => {
  const cols = findAll((e) => e.tagName === 'td', [row]).map((x) =>
    textContent(x),
  )
  return { date: fParseDate(cols[0]), value: fParseFloat(cols[1]) }
}

const fParseFloat = (x: string) => {
  const value = parseFloat(x)
  assert(!isNaN(value))
  return value
}

const fParseDate = (x: string) => {
  const date = Date.parse(x)
  assert(!isNaN(date))
  return date
}
