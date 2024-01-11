import { ValueForMonthRange, ValueForMonthRanges } from '@tpaw/common'
import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { SimpleRange } from '../../Utils/SimpleRange'
import { PlanParamsExtended } from '../ExtentPlanParams'

export function planParamsProcessByMonthParams(
  planParamsExt: PlanParamsExtended,
  monthlyInflation: number,
) {
  const { asMFN, withdrawalStartMonth, numMonths, planParams } = planParamsExt
  const {
    wealth,
    adjustmentsToSpending: { extraSpending },
  } = planParams
  const withdrawalStart = asMFN(withdrawalStartMonth)
  const lastWorkingMonth = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endMonth = numMonths - 1

  const fromValueForMonthRange = (
    { monthRange, value, nominal }: ValueForMonthRange,
    minMonth: number,
    maxMonth: number,
  ) => {
    const values = new Float64Array(numMonths)

    const normMonthRange = asMFN(monthRange)
    if (maxMonth < normMonthRange.start) return { values, validRange: null }
    const start = Math.max(normMonthRange.start, minMonth)
    if (normMonthRange.end < start) return { values, validRange: null }
    const end = Math.min(normMonthRange.end, maxMonth)
    _.range(start, end + 1).forEach(
      (monthsFromNow) =>
        (values[monthsFromNow] = nominalToReal({
          value: { value, nominal },
          monthlyInflation,
          monthsFromNow,
        })),
    )
    return { values, validRange: { start, end } }
  }

  const fromValueForMonthRanges = (
    valueForMonthRanges: ValueForMonthRanges,
    minMonth: number,
    maxMonth: number,
  ) => {
    const byId: Record<
      string,
      { values: Float64Array; validRange: SimpleRange | null }
    > = _.mapValues(valueForMonthRanges, (x) =>
      fromValueForMonthRange(x, minMonth, maxMonth),
    )
    const total = new Float64Array(numMonths)
    const parts = _.values(byId)
    total.forEach((__, i) => (total[i] = _.sum(parts.map((x) => x.values[i]))))
    return { byId, total }
  }

  const result = {
    wealth: (() => {
      const futureSavings = fromValueForMonthRanges(
        wealth.futureSavings,
        0,
        lastWorkingMonth,
      )
      const incomeDuringRetirement = fromValueForMonthRanges(
        wealth.incomeDuringRetirement,
        withdrawalStart,
        endMonth,
      )
      const total = new Float64Array(numMonths)
      total.forEach(
        (__, i) =>
          (total[i] = futureSavings.total[i] + incomeDuringRetirement.total[i]),
      )
      return { total, futureSavings, incomeDuringRetirement }
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
    risk: {
      tpawAndSPAW: {
        lmp: (() => {
          const result = new Float64Array(numMonths)
          _.times(numMonths, (month) => {
            result[month] =
              month < withdrawalStart ? 0 : planParams.risk.tpawAndSPAW.lmp
          })
          return result
        })(),
      },
    },
  }

  return result
}