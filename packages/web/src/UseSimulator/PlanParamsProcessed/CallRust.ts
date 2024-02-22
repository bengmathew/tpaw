import { PlanParams, block } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import { fWASM } from '../Simulator/GetWASM'
import { PlanParamsNormalized } from '../NormalizePlanParams'
import { Record } from '../../Utils/Record'

export namespace CallRust {
  export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
  export const processPlanParams = (
    planParamsNorm: PlanParamsNormalized,
    marketData: Rust.DataForMarketBasedPlanParamValues,
  ) => {
    const wasm = fWASM()
    let start = performance.now()
    
    const fromRust = wasm.process_plan_params(planParamsNorm, marketData)

    // console.log('process_plan_params:part1:', performance.now() - start)
    start = performance.now()

    const withoutArrays = JSON.parse(
      fromRust.without_arrays(),
    ) as Rust.PlanParamsProcessed
    const result = {
      ...withoutArrays,
      historicalMonthlyReturnsAdjusted: {
        stocks: {
          ...withoutArrays.historicalMonthlyReturnsAdjusted.stocks,
          logSeries: fromRust
            .array({
              type: 'historicalMonthlyReturnsAdjusted.stocks.logSeries',
            })
            .slice(),
        },
        bonds: {
          ...withoutArrays.historicalMonthlyReturnsAdjusted.bonds,
          logSeries: fromRust
            .array({ type: 'historicalMonthlyReturnsAdjusted.bonds.logSeries' })
            .slice(),
        },
      },
      byMonth: block(() => {
        const forToValueForMonthRanges = (
          x: typeof withoutArrays.byMonth.wealth.futureSavings,
          t:
            | 'wealth.futureSavings'
            | 'wealth.incomeDuringRetirement'
            | 'adjustmentsToSpending.extraSpending.essential'
            | 'adjustmentsToSpending.extraSpending.discretionary',
        ) => ({
          ...x,
          byId: Record.mapValues(x.byId, (v, id) => ({
            ...v,
            values: fromRust.array({ type: `${t}.byId`, id }).slice(),
          })),
          total: fromRust.array({ type: `${t}.total` }).slice(),
        })
        return {
          ...withoutArrays.byMonth,
          wealth: {
            ...withoutArrays.byMonth.wealth,
            futureSavings: forToValueForMonthRanges(
              withoutArrays.byMonth.wealth.futureSavings,
              'wealth.futureSavings',
            ),
            incomeDuringRetirement: forToValueForMonthRanges(
              withoutArrays.byMonth.wealth.incomeDuringRetirement,
              'wealth.incomeDuringRetirement',
            ),
            total: fromRust.array({ type: 'wealth.total' }).slice(),
          },
          adjustmentsToSpending: {
            ...withoutArrays.byMonth.adjustmentsToSpending,
            extraSpending: {
              ...withoutArrays.byMonth.adjustmentsToSpending.extraSpending,
              essential: forToValueForMonthRanges(
                withoutArrays.byMonth.adjustmentsToSpending.extraSpending
                  .essential,
                'adjustmentsToSpending.extraSpending.essential',
              ),
              discretionary: forToValueForMonthRanges(
                withoutArrays.byMonth.adjustmentsToSpending.extraSpending
                  .discretionary,
                'adjustmentsToSpending.extraSpending.discretionary',
              ),
            },
          },
          risk: {
            ...withoutArrays.byMonth.risk,
            tpawAndSPAW: {
              ...withoutArrays.byMonth.risk.tpawAndSPAW,
              lmp: fromRust.array({ type: 'risk.tpawAndSPAW.lmp' }).slice(),
            },
          },
        }
      }),
    }
    fromRust.free()
    // console.log('process_plan_params:part2:', performance.now() - start)
    start = performance.now()

    return result
  }
}

const ___ = (rs: Rust.PlanParamsNormalized, ts: PlanParamsNormalized) => {
  rs = ts
  ts = rs
}
