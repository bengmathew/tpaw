import _ from 'lodash'
import { blendReturns } from '../../Utils/BlendReturns'
import { getNetPresentValue } from '../../Utils/GetNetPresentValue'

import { annualToMonthlyReturnRate } from '@tpaw/common'
import { CallRust } from './CallRust'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'
import * as Rust from '@tpaw/simulator'
import { Record } from '../../Utils/Record'

export const planParamsProcessNetPresentValue = (
  numMonths: number,
  risk: ReturnType<typeof planParamsProcessRisk>,
  legacyTarget: number,
  byMonth: ReturnType<(typeof CallRust)['processPlanParams']>['byMonth'],
  expectedReturnsForPlanning: CallRust.PlanParamsProcessed['expectedReturnsForPlanning'],
) => {
  const monthlyExpectedReturns = annualToMonthlyReturnRate({
    stocks:
      expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.stocks.value,
    bonds:
      expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.bonds.value,
  })

  const bondRateArr = _.times(
    byMonth.wealth.total.length,
    () => monthlyExpectedReturns.bonds,
  )
  const regularReturns = blendReturns(monthlyExpectedReturns)
  const regularRateArr = risk.tpaw.allocation.map((x) =>
    regularReturns({ stocks: x }),
  )

  const _calcObj = (
    x: {
      total: Float64Array
      byId: Record<string, { values: Float64Array }>
    },
    rate: number[],
  ) => ({
    byId: _.mapValues(x.byId, (x) =>
      Record.map(getNetPresentValue(rate, x.values), (k, v) => [k, v[0]]),
    ),
  })

  return {
    tpaw: {
      wealth: {
        futureSavings: _calcObj(byMonth.wealth.futureSavings, bondRateArr),
        incomeDuringRetirement: _calcObj(
          byMonth.wealth.incomeDuringRetirement,
          bondRateArr,
        ),
      },
      adjustmentsToSpending: {
        extraSpending: {
          essential: _calcObj(
            byMonth.adjustmentsToSpending.extraSpending.essential,
            bondRateArr,
          ),
          discretionary: _calcObj(
            byMonth.adjustmentsToSpending.extraSpending.discretionary,
            regularRateArr,
          ),
        },
        legacy:
          legacyTarget /
          Math.pow(
            1.0 + regularReturns(risk.tpaw.allocationForLegacy),
            numMonths,
          ),
      },
    },
  }
}
