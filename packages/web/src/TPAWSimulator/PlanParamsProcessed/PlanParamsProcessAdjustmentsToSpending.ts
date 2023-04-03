import _ from 'lodash'
import { nominalToReal } from '../../Utils/NominalToReal'
import { ParamsExtended } from '../ExtentParams'

export function planParamsProcessAdjustmentsToSpending(
  paramsExt: ParamsExtended,
  monthlyInflation: number,
) {
  const { params, numMonths } = paramsExt
  return {
    tpawAndSPAW: (() => {
      const { monthlySpendingCeiling, monthlySpendingFloor, legacy } =
        params.plan.adjustmentsToSpending.tpawAndSPAW
      return {
        monthlySpendingCeiling:
          monthlySpendingCeiling === null
            ? null
            : Math.max(monthlySpendingCeiling, params.plan.risk.tpawAndSPAW.lmp),
        monthlySpendingFloor,
        legacy: (() => {
          const { total } = params.plan.adjustmentsToSpending.tpawAndSPAW.legacy
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
