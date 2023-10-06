export type RunSimulationInWASMResult = {
  byMonthsFromNowByRun: {
    savingsPortfolio: {
      start: { balance: Float64Array[] }
      withdrawals: {
        essential: Float64Array[]
        discretionary: Float64Array[]
        regular: Float64Array[]
        total: Float64Array[]
        fromSavingsPortfolioRate: Float64Array[]
      }
      afterWithdrawals: {
        allocation: { stocks: Float64Array[] }
      }
    }
    totalPortfolio: {
      afterWithdrawals: {
        allocation: { stocks: Float64Array[] }
      }
    }
  }
  byRun: {
    numInsufficientFundMonths: Int32Array
    endingBalanceOfSavingsPortfolio: Float64Array
  }
  annualStatsForSampledReturns: Record<
    'stocks' | 'bonds',
    { n: number } & Record<
      'ofBase' | 'ofLog',
      {
        mean: number
        variance: number
        standardDeviation: number
        n: number
      }
    >
  >
  perf: [
    ['runs', number],
    ['post', number],
    ['rest', number],
    ['total', number],
  ]
}
