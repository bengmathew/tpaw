import { assert, block, letIn, noCase } from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { NormalizedGlidePath } from '../NormalizePlanParams/NormalizeGlidePath'
import { NormalizedMonthRange } from '../NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import { NormalizedLabeledAmountTimed } from '../NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { fWASM } from '../Simulator/GetWASM'

export namespace CallRust {
  export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
  export const processPlanParams = (
    planParamsNorm: PlanParamsNormalized,
    marketData: Rust.DataForMarketBasedPlanParamValues,
  ) => {
    const wasm = fWASM()
    let start = performance.now()

    const planParamsRust = getPlanParamsRust(planParamsNorm)
    const fromRust = wasm.process_plan_params(planParamsRust, marketData)

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
        const forLabeledAmountTimedList = (
          x: typeof withoutArrays.byMonth.wealth.futureSavings,
          t:
            | 'wealth.futureSavings'
            | 'wealth.incomeDuringRetirement'
            | 'adjustmentsToSpending.extraSpending.essential'
            | 'adjustmentsToSpending.extraSpending.discretionary',
        ) => ({
          ...x,
          byId: x.byId.map((v) => ({
            ...v,
            values: fromRust.array({ type: `${t}.byId`, id: v.id }).slice(),
          })),
          total: fromRust.array({ type: `${t}.total` }).slice(),
        })
        return {
          ...withoutArrays.byMonth,
          wealth: {
            ...withoutArrays.byMonth.wealth,
            futureSavings: forLabeledAmountTimedList(
              withoutArrays.byMonth.wealth.futureSavings,
              'wealth.futureSavings',
            ),
            incomeDuringRetirement: forLabeledAmountTimedList(
              withoutArrays.byMonth.wealth.incomeDuringRetirement,
              'wealth.incomeDuringRetirement',
            ),
            total: fromRust.array({ type: 'wealth.total' }).slice(),
          },
          adjustmentsToSpending: {
            ...withoutArrays.byMonth.adjustmentsToSpending,
            extraSpending: {
              ...withoutArrays.byMonth.adjustmentsToSpending.extraSpending,
              essential: forLabeledAmountTimedList(
                withoutArrays.byMonth.adjustmentsToSpending.extraSpending
                  .essential,
                'adjustmentsToSpending.extraSpending.essential',
              ),
              discretionary: forLabeledAmountTimedList(
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

  export const getPlanParamsRust = (
    planParamsNorm: PlanParamsNormalized,
  ): Rust.PlanParamsRust => {
    const forMonthRange = (
      monthRange: NormalizedMonthRange,
    ): SimpleRange | null => {
      switch (monthRange.type) {
        case 'startAndEnd':
          return monthRange.end.isInThePast
            ? null
            : {
                start: monthRange.start.asMFN,
                end: monthRange.end.asMFN,
              }
        case 'startAndDuration':
          return {
            start: monthRange.start.asMFN,
            end: monthRange.duration.asMFN,
          }
        case 'endAndDuration':
          return {
            start: monthRange.duration.asMFN,
            end: monthRange.end.asMFN,
          }
        default:
          noCase(monthRange)
      }
    }

    const forAmountAndTiming = (
      amountAndTiming: NormalizedLabeledAmountTimed['amountAndTiming'],
    ): Rust.AmountAndTiming | null => {
      switch (amountAndTiming.type) {
        case 'inThePast':
          return null
        case 'oneTime':
          return {
            type: 'oneTime',
            amount: amountAndTiming.amount,
            month: amountAndTiming.month.asMFN,
          }
        case 'recurring':
          assert(amountAndTiming.delta === null)
          assert(amountAndTiming.everyXMonths === 1)
          return {
            type: 'recurring',
            baseAmount: amountAndTiming.baseAmount,
            validMonthRange: amountAndTiming.monthRange.validRangeAsMFN,
            monthRange: forMonthRange(amountAndTiming.monthRange),
          }
      }
    }

    const forLabeledAmountTimed = (
      valueForMonthRange: NormalizedLabeledAmountTimed,
    ): Rust.LabeledAmountTimed | null => {
      const amountAndTiming = forAmountAndTiming(
        valueForMonthRange.amountAndTiming,
      )
      if (amountAndTiming === null) return null
      return {
        id: valueForMonthRange.id,
        nominal: valueForMonthRange.nominal,
        amountAndTiming,
      }
    }

    const forLabeledAmountTimedList = (
      valueForMonthRanges: NormalizedLabeledAmountTimed[],
    ): Rust.LabeledAmountTimed[] =>
      _.compact(valueForMonthRanges.map(forLabeledAmountTimed))

    const forGlidePath = (
      valueForMonthRange: NormalizedGlidePath,
    ): Rust.GlidePath => ({
      now: { stocks: valueForMonthRange.now.stocks },
      intermediate: valueForMonthRange.intermediate
        .filter((x) => !x.ignore)
        .map((x) => ({
          month: x.month.asMFN,
          stocks: x.stocks,
        })),
      end: { stocks: valueForMonthRange.end.stocks },
    })

    const ages: Rust.PlanParamsRust['ages'] = {
      simulationMonths: letIn(planParamsNorm.ages, ({ simulationMonths }) => ({
        numMonths: simulationMonths.numMonths,
        lastMonthAsMFN: simulationMonths.lastMonthAsMFN,
        withdrawalStartMonthAsMFN: simulationMonths.withdrawalStartMonth.asMFN,
      })),
    }

    const wealth: Rust.PlanParamsRust['wealth'] = {
      futureSavings: forLabeledAmountTimedList(
        planParamsNorm.wealth.futureSavings,
      ),
      incomeDuringRetirement: forLabeledAmountTimedList(
        planParamsNorm.wealth.incomeDuringRetirement,
      ),
    }

    const adjustmentsToSpending: Rust.PlanParamsRust['adjustmentsToSpending'] =
      {
        extraSpending: {
          essential: forLabeledAmountTimedList(
            planParamsNorm.adjustmentsToSpending.extraSpending.essential,
          ),
          discretionary: forLabeledAmountTimedList(
            planParamsNorm.adjustmentsToSpending.extraSpending.discretionary,
          ),
        },
        tpawAndSPAW: {
          monthlySpendingCeiling:
            planParamsNorm.adjustmentsToSpending.tpawAndSPAW
              .monthlySpendingCeiling,
          monthlySpendingFloor:
            planParamsNorm.adjustmentsToSpending.tpawAndSPAW
              .monthlySpendingFloor,
          legacy: {
            total:
              planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy.total,
            external:
              planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy.external,
          },
        },
      }

    const risk = letIn(
      planParamsNorm.risk,
      ({
        tpaw,
        tpawAndSPAW,
        spaw,
        spawAndSWR,
        swr,
      }): Rust.PlanParamsRust['risk'] => ({
        tpaw: {
          riskTolerance: {
            at20: tpaw.riskTolerance.at20,
            deltaAtMaxAge: tpaw.riskTolerance.deltaAtMaxAge,
            forLegacyAsDeltaFromAt20:
              tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
          },
          timePreference: tpaw.timePreference,
          additionalAnnualSpendingTilt: tpaw.additionalAnnualSpendingTilt,
        },
        tpawAndSPAW: {
          lmp: tpawAndSPAW.lmp,
        },
        spaw: {
          annualSpendingTilt: spaw.annualSpendingTilt,
        },
        spawAndSWR: {
          allocation: forGlidePath(spawAndSWR.allocation),
        },
        swr: swr,
      }),
    )

    const advanced = letIn(
      planParamsNorm.advanced,
      (advanced): Rust.PlanParamsRust['advanced'] => ({
        expectedReturnsForPlanning: block(() => {
          switch (advanced.expectedReturnsForPlanning.type) {
            case 'regressionPrediction,20YearTIPSYield':
            case 'conservativeEstimate,20YearTIPSYield':
            case '1/CAPE,20YearTIPSYield':
            case 'historical':
              return { type: advanced.expectedReturnsForPlanning.type }
            case 'manual':
              return {
                type: 'manual',
                stocks: advanced.expectedReturnsForPlanning.stocks,
                bonds: advanced.expectedReturnsForPlanning.bonds,
              }
          }
        }),
        historicalMonthlyLogReturnsAdjustment: block(() => {
          return {
            standardDeviation: {
              stocks: {
                scale:
                  advanced.historicalMonthlyLogReturnsAdjustment
                    .standardDeviation.stocks.scale,
              },
              bonds: {
                enableVolatility:
                  advanced.historicalMonthlyLogReturnsAdjustment
                    .standardDeviation.bonds.enableVolatility,
              },
            },
            overrideToFixedForTesting:
              advanced.historicalMonthlyLogReturnsAdjustment
                .overrideToFixedForTesting,
          }
        }),
        sampling: {
          type: advanced.sampling.type,
          forMonteCarlo: {
            blockSize: advanced.sampling.forMonteCarlo.blockSize.inMonths,
            staggerRunStarts: advanced.sampling.forMonteCarlo.staggerRunStarts,
          },
        },
        annualInflation: block(() => {
          switch (advanced.annualInflation.type) {
            case 'suggested':
              return { type: advanced.annualInflation.type }
            case 'manual':
              return { type: 'manual', value: advanced.annualInflation.value }
          }
        }),
        strategy: advanced.strategy,
      }),
    )

    return {
      ages,
      wealth,
      adjustmentsToSpending,
      risk,
      advanced,
    }
  }
}
