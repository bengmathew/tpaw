import _ from 'lodash'
import { blendReturns } from '../../Utils/BlendReturns'
import { getNetPresentValue } from '../../Utils/GetNetPresentValue'

import { Record } from '../../Utils/Record'
import { CallRust } from './CallRust'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

export const planParamsProcessNetPresentValue = (
  numMonths: number,
  risk: ReturnType<typeof planParamsProcessRisk>,
  legacyTarget: number,
  byMonth: ReturnType<(typeof CallRust)['processPlanParams']>['byMonth'],
  returnsStatsForPlanning: CallRust.PlanParamsProcessed['returnsStatsForPlanning'],
) => {
  const monthlyExpectedReturns = {
    stocks: returnsStatsForPlanning.stocks.empiricalMonthlyNonLogExpectedReturn,
    bonds: returnsStatsForPlanning.bonds.empiricalMonthlyNonLogExpectedReturn,
  }

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
      total: number[]
      byId: { id: string; values: number[] }[]
    },
    rate: number[],
  ) => ({
    byId: x.byId.map((x) => ({
      id: x.id,
      values: Record.mapValues(getNetPresentValue(rate, x.values), (x) => x[0]),
    })),
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
