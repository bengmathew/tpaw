import _ from 'lodash'
import { blendReturns } from '../../Utils/BlendReturns'
import { getNetPresentValue } from '../../Utils/GetNetPresentValue'

import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'
import { PlanParamsExtended } from '../ExtentPlanParams'

// TODO: Should we get this from wasm?
export const planParamsProcessNetPresentValue = (
  paramsExt: PlanParamsExtended,
  risk: ReturnType<typeof planParamsProcessRisk>,
  legacyTarget: number,
  byMonth: ReturnType<typeof planParamsProcessByMonthParams>,
  monthlyExpectedReturns: { stocks: number; bonds: number },
) => {
  const { numMonths } = paramsExt
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
      byId: Record<string, Float64Array>
    },
    rate: number[],
  ) => ({
    byId: _.mapValues(x.byId, (x) => getNetPresentValue(rate, x)),
    total: getNetPresentValue(rate, x.total),
  })

  return {
    tpaw: {
      wealth: {
        total: getNetPresentValue(
          bondRateArr,
          byMonth.wealth.total,
        ),
        futureSavings: _calcObj(
          byMonth.wealth.futureSavings,
          bondRateArr,
        ),
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
            1.0 +
              blendReturns(monthlyExpectedReturns)(
                risk.tpaw.allocationForLegacy,
              ),
            numMonths,
          ),
      },
    },
  }
}
