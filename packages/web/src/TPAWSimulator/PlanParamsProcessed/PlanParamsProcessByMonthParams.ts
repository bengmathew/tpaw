import { ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { ParamsExtended } from '../ExtentParams'

export function planParamsProcessByMonthParams(
  paramsExt: ParamsExtended,
  monthlyInflation: number,
) {
  const { asMFN, withdrawalStartMonth, numMonths, params } = paramsExt
  const {
    wealth,
    adjustmentsToSpending: { extraSpending },
  } = params.plan
  const withdrawalStart = asMFN(withdrawalStartMonth)
  const lastWorkingMonth = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endMonth = numMonths - 1

  const fromValueForMonthRange = (
    { monthRange, value, nominal }: ValueForMonthRange,
    minMonth: number,
    maxMonth: number,
  ) => {
    const result = new Float64Array(numMonths)
    const normMonthRange = asMFN(monthRange)
    const start = _.clamp(normMonthRange.start, minMonth, maxMonth)
    if (normMonthRange.end < start) return result
    const end = Math.min(normMonthRange.end, maxMonth)
    _.range(start, end + 1).forEach(
      (monthsFromNow) =>
        (result[monthsFromNow] = nominalToReal({
          value: { value, nominal },
          monthlyInflation,
          monthsFromNow,
        })),
    )
    return result
  }

  const fromValueForMonthRanges = (
    valueForMonthRanges: ValueForMonthRange[],
    minMonth: number,
    maxMonth: number,
  ) => {
    const byId: Record<number, Float64Array> = _.fromPairs(
      valueForMonthRanges.map((x) => [
        x.id,
        fromValueForMonthRange(x, minMonth, maxMonth),
      ]),
    )
    const total = new Float64Array(numMonths)
    const parts = _.values(byId)
    total.forEach((__, i) => (total[i] = _.sum(parts.map((x) => x[i]))))
    return { byId, total }
  }

  const result = {
    // TODO: Rename to wealth
    futureSavingsAndRetirementIncome: (() => {
      const futureSavings = fromValueForMonthRanges(
        wealth.futureSavings,
        0,
        lastWorkingMonth,
      )
      const retirementIncome = fromValueForMonthRanges(
        wealth.retirementIncome,
        withdrawalStart,
        endMonth,
      )
      const total = new Float64Array(numMonths)
      total.forEach(
        (__, i) =>
          (total[i] = futureSavings.total[i] + retirementIncome.total[i]),
      )
      return { total, futureSavings, retirementIncome }
    })(),
    adjustmentsToSpending: {
      extraSpending: {
        essential: fromValueForMonthRanges(
          extraSpending.essential,
          0,
          endMonth,
        ),
        discretionary: fromValueForMonthRanges(
          extraSpending.discretionary,
          0,
          endMonth,
        ),
      },
    },
    // TODO: rename to risk
    tpawAndSPAW: {
      risk: {
        lmp: (() => {
          const result = new Float64Array(numMonths)
          _.times(numMonths, (month) => {
            result[month] =
              month < withdrawalStart ? 0 : params.plan.risk.tpawAndSPAW.lmp
          })
          return result
        })(),
      },
    },
  }

  return result
}
