import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { PlanParamsExt } from '../PlanParamsExt'

export function planParamsProcessAdjustmentsToSpending(
  paramsExt: PlanParamsExt,
  monthlyInflation: number,
) {
  const { params, numMonths } = paramsExt
  return {
    tpawAndSPAW: (() => {
      const { monthlySpendingCeiling, monthlySpendingFloor, legacy } =
        params.adjustmentsToSpending.tpawAndSPAW
      return {
        monthlySpendingCeiling:
          monthlySpendingCeiling === null
            ? null
            : Math.max(monthlySpendingCeiling, params.risk.tpawAndSPAW.lmp),
        monthlySpendingFloor,
        legacy: (() => {
          const { total } = params.adjustmentsToSpending.tpawAndSPAW.legacy
          const external = _.sum(
            legacy.external.map((x) =>
              nominalToReal({
                value: x,
                monthlyInflation,
                monthsFromNow: numMonths,
              }),
            ),
          )
          const target = Math.max(total - external, 0)
          return { total, external, target }
        })(),
      }
    })(),
  }
}
