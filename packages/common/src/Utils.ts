import _ from 'lodash'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function assert(condition: any): asserts condition {
  if (!condition) {
    throw new Error('Assert')
  }
}

export function assertFalse(): never {
  throw new Error('Assert')
}

export function fGet<T>(x: T | null | undefined): T {
  assert(x !== null && x !== undefined)
  return x as T
}

export function noCase(x: never): never {
  // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
  throw new Error(`Didn't expect to get here: ${x}`)
}

export function nundef<T>(x: T | undefined): T {
  assert(x !== undefined)
  return x as T
}

export const linearFnFomPoints = (
  x0: number,
  y0: number,
  x1: number,
  y1: number,
) => {
  const slope = (y1 - y0) / (x1 - x0)
  return linearFnFromPointAndSlope(x0, y0, slope)
}

export const linearFnFromPointAndSlope = (
  x: number,
  y: number,
  slope: number,
) => {
  const intercept = y - slope * x
  return linearFnFromSlopeAndIntercept(slope, intercept)
}

export type LinearFn = ReturnType<typeof linearFnFromSlopeAndIntercept>
export const linearFnFromSlopeAndIntercept = (
  slope: number,
  intercept: number,
) => {
  const result = (x: number) => slope * x + intercept
  result.inverse = (y: number) => (y - intercept) / slope
  return result
}

export const preciseRange = (
  start: number,
  end: number,
  stepSize: number,
  precision: number,
) => {
  return _.range(_.round((end - start) / stepSize + 1)).map((x) =>
    _.round(x * stepSize + start, precision),
  )
}

export function annualToMonthlyReturnRate(
  annual: number,
  // correction: number,
): number
export function annualToMonthlyReturnRate(
  annual: {
    stocks: number
    bonds: number
  },
  // correction: number,
): { stocks: number; bonds: number }
export function annualToMonthlyReturnRate(
  annual: number | { stocks: number; bonds: number },
  // correction: number,
): number | { stocks: number; bonds: number } {
  return typeof annual === 'number'
    ? Math.pow(1 + annual, 1 / 12) - 1
    : {
        stocks: annualToMonthlyReturnRate(annual.stocks),
        bonds: annualToMonthlyReturnRate(annual.bonds),
      }
}

export function monthlyToAnnualReturnRate(monthly: number): number
export function monthlyToAnnualReturnRate(monthly: {
  stocks: number
  bonds: number
}): { stocks: number; bonds: number }
export function monthlyToAnnualReturnRate(
  monthly: number | { stocks: number; bonds: number },
): number | { stocks: number; bonds: number } {
  return typeof monthly === 'number'
    ? Math.pow(1 + monthly, 12) - 1
    : {
        stocks: monthlyToAnnualReturnRate(monthly.stocks),
        bonds: monthlyToAnnualReturnRate(monthly.bonds),
      }
}

export const getLogReturns = (returns: number[]) =>
  returns.map((x) => Math.log(1 + x))

export function getStats<T extends Float64Array | number[]>(returns: T) {
  const expectedValue = _.mean(returns)
  const variance =
    _.sumBy(returns, (x) => Math.pow(x - expectedValue, 2)) /
    (returns.length - 1)
  const standardDeviation = Math.sqrt(variance)

  return {
    returns,
    expectedValue,
    mean: expectedValue,
    variance,
    standardDeviation,
  }
}

export function getStatsWithLog(x: number[]) {
  return {
    ofBase: getStats(x),
    ofLog: getStats(getLogReturns(x)),
  }
}

export const sequentialAnnualReturnsFromMonthly = (monthly: number[]) =>
  blockify(monthly, 12).map(monthRateArrToYear)

export const blockify = (x: number[], blockSize: number) => {
  const numBlocks = x.length + 1 - blockSize
  assert(numBlocks > 0)
  return _.range(0, numBlocks).map((i) => x.slice(i, i + blockSize))
}

export const monthRateArrToYear = (year: number[]) =>
  year.map((x) => 1 + x).reduce((x, y) => x * y, 1) - 1

export const block = <T>(fn: () => T): T => fn()

export const letIn = <Vars, U>(x: Vars, fn: (vars: Vars) => U): U => fn(x)

export const generateRandomString = (length: number) => {
  let result = ''
  const characters = 'abcdefghijklmnopqrstuvwxyz'
  const charactersLength = characters.length
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength))
  }
  return result
}

export const generateSmallId = () => generateRandomString(10)
