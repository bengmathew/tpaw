import _ from 'lodash'
import { blendReturns } from '../../Utils/BlendReturns'
import { getNetPresentValue } from '../../Utils/GetNetPresentValue'
import { ParamsExtended } from '../ExtentParams'
import { planParamsProcessByMonthParams } from './PlanParamsProcessByMonthParams'
import { planParamsProcessRisk } from './PlanParamsProcessRisk'

// TODO: Should we get this from wasm?
export const planParamsProcessNetPresentValue = (
  paramsExt: ParamsExtended,
  risk: ReturnType<typeof planParamsProcessRisk>,
  legacyTarget: number,
  byMonth: ReturnType<typeof planParamsProcessByMonthParams>,
  monthlyExpectedReturns: { stocks: number; bonds: number },
) => {
  const { numMonths } = paramsExt
  const bondRateArr = _.times(
    byMonth.futureSavingsAndRetirementIncome.total.length,
    () => monthlyExpectedReturns.bonds,
  )
  const regularReturns = blendReturns(monthlyExpectedReturns)
  const regularRateArr = risk.tpaw.allocation.map((x) =>
    regularReturns({ stocks: x }),
  )

  const _calcObj = (
    x: {
      total: Float64Array
      byId: Record<number, Float64Array>
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
          byMonth.futureSavingsAndRetirementIncome.total,
        ),
        futureSavings: _calcObj(
          byMonth.futureSavingsAndRetirementIncome.futureSavings,
          bondRateArr,
        ),
        retirementIncome: _calcObj(
          byMonth.futureSavingsAndRetirementIncome.retirementIncome,
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
