import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'

export function planParamsProcessAdjustmentsToSpending(
  planParamsNorm: PlanParamsNormalized,
  monthlyInflation: number,
) {
  return {
    tpawAndSPAW: (() => {
      const { monthlySpendingCeiling, monthlySpendingFloor, legacy } =
        planParamsNorm.adjustmentsToSpending.tpawAndSPAW
      return {
        monthlySpendingCeiling:
          monthlySpendingCeiling === null
            ? null
            : Math.max(monthlySpendingCeiling, planParamsNorm.risk.tpawAndSPAW.lmp),
        monthlySpendingFloor,
        legacy: (() => {
          const { total } = planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy
          const external = _.sum(
            legacy.external.map((x) =>
              nominalToReal({
                value: x,
                monthlyInflation,
                monthsFromNow: planParamsNorm.ages.simulationMonths.numMonths,
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
