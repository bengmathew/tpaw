import {TPAWParams} from './TPAWParams'

export function getDefaultParams() {
  const params: TPAWParams = {
    v: 9,
    strategy: 'TPAW',
    people: {
      withPartner: false,
      person1: {
        displayName: null,
        ages: {type: 'notRetired', current: 35, retirement: 65, max: 100},
      },
    },
    returns: {
      expected: {stocks: 0.042, bonds: 0.008},
      historical: {type: 'default', adjust: {type: 'toExpected'}},
    },
    inflation: 0.023,
    targetAllocation: {
      regularPortfolio: {
        forTPAW: {stocks: 0.35},
        forSPAWAndSWR: {
          start: {stocks: 0.5},
          intermediate: [],
          end: {stocks: 0.5},
        },
      },
      legacyPortfolio: {stocks: 0.7},
    },
    swrWithdrawal: {type: 'asPercent', percent: 0.04},
    scheduledWithdrawalGrowthRate: 0.005,
    savingsAtStartOfStartYear: 100000,
    savings: [],
    retirementIncome: [],
    withdrawals: {
      lmp: 0,
      essential: [],
      discretionary: [],
    },
    spendingCeiling: null,
    spendingFloor: null,
    legacy: {
      total: 0,
      external: [],
    },
    sampling: 'monteCarlo',
    display: {
      alwaysShowAllYears: false,
    },
  }

  return params
}
