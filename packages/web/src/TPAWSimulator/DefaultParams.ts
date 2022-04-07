import { TPAWParams } from './TPAWParams'

export function getDefaultParams() {
  const params: TPAWParams = {
    v: 5,
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 35, retirement: 65, max: 100},
      },
    },
    returns: {
      expected: {stocks: 0.035, bonds: -0.005},
      historical: {adjust: {type: 'to', stocks: 0.035, bonds: -0.005}},
    },
    inflation: 0.02,
    targetAllocation: {
      regularPortfolio: {stocks: 0.35},
      legacyPortfolio: {stocks: 0.7},
    },
    scheduledWithdrawalGrowthRate: 0.005,
    savingsAtStartOfStartYear: 100000,
    savings: [],
    retirementIncome: [],
    withdrawals: {
      fundedByBonds: [],
      fundedByRiskPortfolio: [],
    },
    spendingCeiling: null,
    spendingFloor: null,
    legacy: {
      total: 0,
      external: [],
    },
  }

  return params
}
