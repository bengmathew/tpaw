 type YearRangeFixedEdge =
  | 'start'
  | 'lastWorkingYear'
  | 'retirement'
  | 'end'

 type YearRangeEdge = number | YearRangeFixedEdge

 type YearRange = {start: YearRangeEdge; end: YearRangeEdge}
 type ValueForYearRange = {
  label: string | null
  yearRange: YearRange
  value: number
  nominal: boolean
}

export type TPAWParamsV3 = {
  v: 3
  age: {
    start: number
    retirement: number
    end: number
  }
  returns: {
    expected: {
      stocks: number
      bonds: number
    }
    historical: {
      adjust:
        | {type: 'by'; stocks: number; bonds: number}
        | {type: 'to'; stocks: number; bonds: number}
    }
  }
  inflation: number
  targetAllocation: {
    regularPortfolio: {stocks: number}
    legacyPortfolio: {stocks: number}
  }
  spendingCeiling: number | null
  spendingFloor: number | null
  scheduledWithdrawalGrowthRate: number
  savingsAtStartOfStartYear: number
  savings: ValueForYearRange[]
  retirementIncome: ValueForYearRange[]
  withdrawals: {
    fundedByBonds: ValueForYearRange[]
    fundedByRiskPortfolio: ValueForYearRange[]
  }
  legacy: {
    total: number
    external: {label: string | null; value: number; nominal: boolean}[]
  }
}
export type TPAWParamsV3WithoutHistorical = Omit<TPAWParamsV3, 'returns'> & {
  returns: Omit<TPAWParamsV3['returns'], 'historical'>
}
