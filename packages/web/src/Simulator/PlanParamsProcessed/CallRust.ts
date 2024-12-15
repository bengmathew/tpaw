import {
  PLAN_PARAMS_CONSTANTS,
  assert,
  block,
  fGet,
  letIn,
  noCase,
} from '@tpaw/common'
import * as Rust from '@tpaw/simulator'
import _ from 'lodash'
import { SimpleRange } from '../../Utils/SimpleRange'
import { NormalizedGlidePath } from '../NormalizePlanParams/NormalizeGlidePath'
import { NormalizedMonthRange } from '../NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import { NormalizedLabeledAmountTimed } from '../NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { fWASM } from '../Simulator/GetWASM'
import { unpack } from 'msgpackr'

// TODO: Testing. Delete (also remove msgpackr from package.json)
export namespace CallRust {
  export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
  export const processPlanParams = (
    planParamsNorm: PlanParamsNormalized,
    marketData: Rust.DataForMarketBasedPlanParamValues,
  ) => {
    const wasm = fWASM()
    let start = performance.now()

    const planParamsRust = getPlanParamsRust(planParamsNorm)
    // console.log('process_plan_params:part1:', performance.now() - start)
    start = performance.now()

    const encoded = wasm.process_plan_params_lib(planParamsRust, marketData)
    // console.log('process_plan_params:part2:', performance.now() - start)
    start = performance.now()

    const planParamsProcessed = unpack(encoded)
    // console.log('process_plan_params:part3:', performance.now() - start)
    start = performance.now()

    return planParamsProcessed as Rust.PlanParamsProcessed
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

    const constants: Rust.PlanParamsRust['constants'] = {
      maxAge: PLAN_PARAMS_CONSTANTS.people.ages.person.maxAge,
      riskToleranceNumIntegerValuesStartingFrom0:
        PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values
          .numIntegerValuesStartingFrom0,
      riskToleranceStartRRA:
        PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.startRRA,
      riskToleranceEndRRA:
        PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.endRRA,
    }
    const ages: Rust.PlanParamsRust['ages'] = {
      simulationMonths: letIn(planParamsNorm.ages, ({ simulationMonths }) => ({
        numMonths: simulationMonths.numMonths,
        numWithdrawalMonths: simulationMonths.numWithdrawalMonths,
        lastMonthAsMFN: simulationMonths.lastMonthAsMFN,
        withdrawalStartMonthAsMFN: simulationMonths.withdrawalStartMonth.asMFN,
      })),
      longerLivedPerson: letIn(
        fGet(planParamsNorm.ages[planParamsNorm.ages.longerLivedPersonType]),
        (longerLivedPerson) => ({
          maxAgeInMonths: longerLivedPerson.maxAge.baseValue.inMonths,
          currentAgeInMonths: longerLivedPerson.currentAgeInfo.inMonths,
        }),
      ),
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
        returnsStatsForPlanning: {
          expectedValue: {
            empiricalAnnualNonLog:
              advanced.returnsStatsForPlanning.expectedValue
                .empiricalAnnualNonLog,
          },
          standardDeviation: advanced.returnsStatsForPlanning.standardDeviation,
        },
        historicalReturnsAdjustment: {
          standardDeviation: {
            bonds: advanced.historicalReturnsAdjustment.standardDeviation.bonds,
          },
          overrideToFixedForTesting: block(
            (): Rust.HistoricalResultsAdjustment_OverrideToFixedForTesting => {
              const value =
                advanced.historicalReturnsAdjustment.overrideToFixedForTesting
              switch (value.type) {
                case 'none':
                  return { type: 'none' }
                case 'useExpectedReturnsForPlanning':
                  return {
                    type: 'toExpectedReturns',
                  }
                case 'manual':
                  return {
                    type: 'manual',
                    stocks: value.stocks,
                    bonds: value.bonds,
                  }
                default:
                  noCase(value)
              }
            },
          ),
        },
        sampling:
          advanced.sampling.type === 'monteCarlo'
            ? {
                type: advanced.sampling.type,
                blockSize: advanced.sampling.data.blockSize.inMonths,
                staggerRunStarts: advanced.sampling.data.staggerRunStarts,
              }
            : {
                type: advanced.sampling.type,
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
      constants,
      ages,
      wealth,
      adjustmentsToSpending,
      risk,
      advanced,
    }
  }
}
