import { ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { extendPlanParams, PlanParamsExt } from '../PlanParamsExt'

export function planParamsProcessByMonthParams(
  paramsExt: PlanParamsExt,
  monthlyInflation: number,
) {
  const { asMFN, withdrawalStartMonth, numMonths, params } = paramsExt
  const {
    wealth,
    adjustmentsToSpending: { extraSpending },
  } = params
  const withdrawalStart = asMFN(withdrawalStartMonth)
  const lastWorkingMonth = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endMonth = numMonths - 1
  const byMonth = _.times(numMonths, (month) => ({
    futureSavingsAndRetirementIncome: 0,
    extraSpending: {
      essential: 0,
      discretionary: 0,
    },
    tpawAndSPAW: {
      risk: { lmp: month < withdrawalStart ? 0 : params.risk.tpawAndSPAW.lmp },
    },
  }))

  const exec = (
    values: ValueForMonthRange[],
    minMonth: number,
    maxMonth: number,
    updater: (target: typeof byMonth[0], value: number) => void,
  ) => {
    values.forEach(({ monthRange, value, nominal }) => {
      const normMonthRange = extendPlanParams(params).asMFN(monthRange)
      const start = _.clamp(normMonthRange.start, minMonth, maxMonth)
      const end = _.clamp(normMonthRange.end, start, maxMonth)
      _.range(start, end + 1).forEach((monthsFromNow) => {
        updater(
          byMonth[monthsFromNow],
          nominalToReal({
            value: { value, nominal },
            monthlyInflation,
            monthsFromNow,
          }),
        )
      })
    })
  }

  exec(
    wealth.futureSavings,
    0,
    lastWorkingMonth,
    (t, v) => (t.futureSavingsAndRetirementIncome += v),
  )
  exec(
    wealth.retirementIncome,
    withdrawalStart,
    endMonth,
    (t, v) => (t.futureSavingsAndRetirementIncome += v),
  )
  exec(
    extraSpending.essential,
    0,
    endMonth,
    (t, v) => (t.extraSpending.essential += v),
  )
  exec(
    extraSpending.discretionary,
    0,
    endMonth,
    (t, v) => (t.extraSpending.discretionary += v),
  )
  return byMonth
}
