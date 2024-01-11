import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { PlanParamsExtended } from '../ExtentPlanParams'

export function planParamsProcessAdjustmentsToSpending(
  planParamsExt: PlanParamsExtended,
  monthlyInflation: number,
) {
  const { planParams, numMonths } = planParamsExt
  return {
    tpawAndSPAW: (() => {
      const { monthlySpendingCeiling, monthlySpendingFloor, legacy } =
        planParams.adjustmentsToSpending.tpawAndSPAW
      return {
        monthlySpendingCeiling:
          monthlySpendingCeiling === null
            ? null
            : Math.max(monthlySpendingCeiling, planParams.risk.tpawAndSPAW.lmp),
        monthlySpendingFloor,
        legacy: (() => {
          const { total } = planParams.adjustmentsToSpending.tpawAndSPAW.legacy
          const external = _.sum(
            _.values(legacy.external).map((x) =>
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
