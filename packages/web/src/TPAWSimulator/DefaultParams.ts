import {TPAWParams} from './TPAWParams'

export function getDefaultParams() {
  const params: TPAWParams = {
    age: {
      start: 35,
      retirement: 65,
      end: 100,
    },
    returns: {
      expected: {
        stocks: 0.035,
        bonds: -0.005,
      },
      historical: {
        adjust: {
          type: 'to',
          stocks: 0.035,
          bonds: -0.005,
        },
      },
    },
    inflation: 0.02,
    targetAllocation: {
      regularPortfolio: {stocks: 0.35},
      legacyPortfolio: {stocks: 0.7},
    },
    scheduledWithdrawalGrowthRate: 0.005,
    savingsAtStartOfStartYear: 50000,
    savings: [
      {
        label: 'Savings',
        yearRange: {start: 'start', end: 'lastWorkingYear'},
        value: 10000,
        nominal: false,
      },
      {
        label: 'Social Security',
        yearRange: {start: 70, end: 'end'},
        value: 20000,
        nominal: false,
      },
    ],
    withdrawals: {
      fundedByBonds: [],
      fundedByRiskPortfolio: [],
    },
    spendingCeiling: null,
    legacy: {
      total: 0,
      external:[]
    },
  }

  return params
}
