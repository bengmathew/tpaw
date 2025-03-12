import {
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  LabeledAmountUntimed,
  PLAN_PARAMS_CONSTANTS,
  assertFalse,
  block,
  fGet,
  letIn,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { NormalizedGlidePath } from '@tpaw/common'
import { NormalizedLabeledAmountTimed } from '@tpaw/common'
import { PlanParamsNormalized } from '@tpaw/common'
import {
  WireGlidePath,
  WireInt32Range,
  WirePlanParamsServer,
  WirePlanParamsServerAmountNotTimed,
  WirePlanParamsServerAmountTimed,
  WirePlanParamsServerExpectedReturnsForPlanning,
  WirePlanParamsServerExpectedReturnsForPlanningCustomBondsBase,
  WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase,
  WirePlanParamsServerStrategy,
} from './Wire/wire_plan_params_server'

export const getPlanParamsServer = (
  planParamsNorm: PlanParamsNormalized,
  numRuns: number,
  seed: number,
): WirePlanParamsServer => {
  const toAmountNotTimed = (
    src: LabeledAmountUntimed,
  ): WirePlanParamsServerAmountNotTimed => ({
    isNominal: src.nominal,
    amount: src.amount,
  })
  const toAmountTimed = (
    src: NormalizedLabeledAmountTimed,
  ): WirePlanParamsServerAmountTimed | null => {
    const noDelta = { $case: 'amount', amount: 0 } as const

    switch (src.amountAndTiming.type) {
      case 'inThePast':
        return null
      case 'oneTime':
        return {
          id: src.id,
          isNominal: src.nominal,
          monthRange: {
            start: src.amountAndTiming.month.asMFN,
            end: src.amountAndTiming.month.asMFN,
          },
          validMonthRange: getPlanParamsServer.getValidMonthRangeForAmountTimed(
            src.amountAndTiming,
          ),
          everyXMonths: 1,
          baseAmount: src.amountAndTiming.amount,
          deltaEveryRecurrence: noDelta,
        }
      case 'recurring':
        let amountAndTiming = src.amountAndTiming
        const monthRange = block((): WireInt32Range | null => {
          switch (amountAndTiming.monthRange.type) {
            case 'startAndEnd': {
              const { start, end } = amountAndTiming.monthRange
              return end.isInThePast || end.asMFN < start.asMFN
                ? null
                : { start: start.asMFN, end: end.asMFN }
            }
            case 'startAndDuration': {
              const { start, duration } = amountAndTiming.monthRange
              return { start: start.asMFN, end: duration.asMFN }
            }
            case 'endAndDuration': {
              const { end, duration } = amountAndTiming.monthRange
              return { start: duration.asMFN, end: end.asMFN }
            }
            default:
              noCase(amountAndTiming.monthRange)
          }
        })
        return {
          id: src.id,
          isNominal: src.nominal,
          monthRange: monthRange ?? undefined,
          validMonthRange: getPlanParamsServer.getValidMonthRangeForAmountTimed(
            src.amountAndTiming,
          ),
          everyXMonths: src.amountAndTiming.everyXMonths,
          baseAmount: src.amountAndTiming.baseAmount,
          deltaEveryRecurrence: block(() => {
            if (!amountAndTiming.delta) return noDelta
            switch (amountAndTiming.delta.by.type) {
              case 'percent':
                // When this is implemented, make sure to adjust the percent
                // for every year which might be the input to every recurrence,
                // which is what he server expects.
                assertFalse()
              default:
                noCase(amountAndTiming.delta.by.type)
            }
          }),
        }
      default:
        noCase(src.amountAndTiming)
    }
  }
  const toAmountTimedList = (src: NormalizedLabeledAmountTimed[]) =>
    _.compact(src.map(toAmountTimed))

  const toGlidePath = (src: NormalizedGlidePath): WireGlidePath => ({
    now: src.now.stocks,
    intermediate: src.intermediate.map((x) => ({
      month: x.month.asMFN,
      value: x.stocks,
    })),
    end: src.end.stocks,
  })

  const evaluationTimestampMs = planParamsNorm.datingInfo.isDated
    ? planParamsNorm.datingInfo.nowAsTimestamp
    : planParamsNorm.datingInfo.nowAsTimestampNominal

  const constants: WirePlanParamsServer['constants'] = {
    riskToleranceNumIntegerValuesStartingFrom0:
      PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values
        .numIntegerValuesStartingFrom0,
    riskToleranceStartRra:
      PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.startRRA,
    riskToleranceEndRra:
      PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.endRRA,
  }

  const ages: WirePlanParamsServer['ages'] = {
    simulationMonths: {
      numMonths: planParamsNorm.ages.simulationMonths.numMonths,
      withdrawalStartMonth:
        planParamsNorm.ages.simulationMonths.withdrawalStartMonth.asMFN,
    },
    longerLivedPerson: letIn(
      fGet(planParamsNorm.ages[planParamsNorm.ages.longerLivedPersonType]),
      (longerLivedPerson) => ({
        maxAge: longerLivedPerson.maxAge.baseValue.inMonths,
        currentAge: longerLivedPerson.currentAgeInfo.inMonths,
      }),
    ),
  }
  const wealth: WirePlanParamsServer['wealth'] = {
    portfolioBalance: block(() => {
      const src = planParamsNorm.wealth.portfolioBalance
      return src.isDatedPlan
        ? src.updatedHere
          ? {
              $case: 'updatedHere',
              updatedHere: src.amount,
            }
          : {
              $case: 'notUpdatedHere',
              notUpdatedHere: {
                updatedAtId: src.updatedAtId,
                updatedTo: src.updatedTo,
                updatedAtTimestampMs: src.updatedAtTimestamp,
              },
            }
        : {
            $case: 'updatedHere',
            updatedHere: src.amount,
          }
    }),
    futureSavings: toAmountTimedList(planParamsNorm.wealth.futureSavings),
    incomeDuringRetirement: toAmountTimedList(
      planParamsNorm.wealth.incomeDuringRetirement,
    ),
  }

  const adjustmentsToSpending: WirePlanParamsServer['adjustmentsToSpending'] =
    block(() => {
      const src = planParamsNorm.adjustmentsToSpending
      return {
        extraSpending: {
          essential: toAmountTimedList(src.extraSpending.essential),
          discretionary: toAmountTimedList(src.extraSpending.discretionary),
        },
        tpawAndSpaw: {
          spendingCeiling:
            src.tpawAndSPAW.monthlySpendingCeiling === null
              ? undefined
              : { value: src.tpawAndSPAW.monthlySpendingCeiling },
          spendingFloor:
            src.tpawAndSPAW.monthlySpendingFloor === null
              ? undefined
              : { value: src.tpawAndSPAW.monthlySpendingFloor },
          legacy: {
            total: src.tpawAndSPAW.legacy.total,
            external: src.tpawAndSPAW.legacy.external.map(toAmountNotTimed),
          },
        },
      }
    })

  const risk: WirePlanParamsServer['risk'] = block(() => {
    const src = planParamsNorm.risk
    return {
      tpaw: {
        riskTolerance: {
          at20: src.tpaw.riskTolerance.at20,
          deltaAtMaxAge: src.tpaw.riskTolerance.deltaAtMaxAge,
          forLegacyAsDeltaFromAt20:
            src.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
        },
        timePreference: src.tpaw.timePreference,
        additionalAnnualSpendingTilt: src.tpaw.additionalAnnualSpendingTilt,
      },
      spaw: {
        annualSpendingTilt: src.spaw.annualSpendingTilt,
      },
      spawAndSwr: {
        stockAllocation: toGlidePath(src.spawAndSWR.allocation),
      },
      swr: {
        withdrawal: block(() => {
          switch (src.swr.withdrawal.type) {
            case 'asAmountPerMonth':
              return {
                $case: 'amountPerMonth',
                amountPerMonth: src.swr.withdrawal.amountPerMonth,
              }
            case 'asPercentPerYear':
              return {
                $case: 'percentPerYear',
                percentPerYear: src.swr.withdrawal.percentPerYear,
              }
            case 'default':
              return {
                $case: 'percentPerYear',
                percentPerYear: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                  planParamsNorm.ages.simulationMonths.numWithdrawalMonths,
                ),
              }
            default:
              noCase(src.swr.withdrawal)
          }
        }),
      },
    }
  })

  const advanced: WirePlanParamsServer['advanced'] = {
    returnStatsForPlanning: {
      expectedValue: {
        empiricalAnnualNonLog: block(
          (): WirePlanParamsServerExpectedReturnsForPlanning['empiricalAnnualNonLog'] => {
            let src =
              planParamsNorm.advanced.returnsStatsForPlanning.expectedValue
                .empiricalAnnualNonLog
            switch (src.type) {
              case 'regressionPrediction,20YearTIPSYield':
                return {
                  $case: 'regressionPrediction20YearTipsYield',
                  regressionPrediction20YearTipsYield: {},
                }
              case 'conservativeEstimate,20YearTIPSYield':
                return {
                  $case: 'conservativeEstimate20YearTipsYield',
                  conservativeEstimate20YearTipsYield: {},
                }
              case '1/CAPE,20YearTIPSYield':
                return {
                  $case: 'oneOverCape20YearTipsYield',
                  oneOverCape20YearTipsYield: {},
                }
              case 'historical':
                return { $case: 'historical', historical: {} }
              case 'fixedEquityPremium':
                return {
                  $case: 'fixedEquityPremium',
                  fixedEquityPremium: src.equityPremium,
                }
              case 'custom':
                return {
                  $case: 'custom',
                  custom: {
                    stocks: {
                      base: block(
                        (): WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase => {
                          switch (src.stocks.base) {
                            case 'regressionPrediction':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase.RegressionPrediction
                            case 'conservativeEstimate':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase.ConservativeEstimate
                            case '1/CAPE':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase.OneOverCape
                            case 'historical':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomStocksBase.HistoricalStocks
                            default:
                              noCase(src.stocks.base)
                          }
                        },
                      ),
                      delta: src.stocks.delta,
                    },
                    bonds: {
                      base: block(
                        (): WirePlanParamsServerExpectedReturnsForPlanningCustomBondsBase => {
                          switch (src.bonds.base) {
                            case '20YearTIPSYield':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomBondsBase.TwentyYearTipsYield
                            case 'historical':
                              return WirePlanParamsServerExpectedReturnsForPlanningCustomBondsBase.HistoricalBonds
                            default:
                              noCase(src.bonds.base)
                          }
                        },
                      ),
                      delta: src.bonds.delta,
                    },
                  },
                }
              case 'fixed':
                return {
                  $case: 'fixed',
                  fixed: { stocks: src.stocks, bonds: src.bonds },
                }

              default:
                noCase(src)
            }
          },
        ),
      },
      standardDeviation: {
        stocks: {
          scale: {
            log: planParamsNorm.advanced.returnsStatsForPlanning
              .standardDeviation.stocks.scale.log,
          },
        },
      },
    },
    historicalReturnsAdjustment: {
      standardDeviation: {
        bonds: {
          scale: {
            log: planParamsNorm.advanced.historicalReturnsAdjustment
              .standardDeviation.bonds.scale.log,
          },
        },
      },
      overrideToFixedForTesting: block(() => {
        const value =
          planParamsNorm.advanced.historicalReturnsAdjustment
            .overrideToFixedForTesting
        switch (value.type) {
          case 'none':
            return { $case: 'none', none: {} }
          case 'useExpectedReturnsForPlanning':
            return {
              $case: 'toExpectedReturnsForPlanning',
              toExpectedReturnsForPlanning: {},
            }
          case 'manual':
            return {
              $case: 'manual',
              manual: {
                stocks: value.stocks,
                bonds: value.bonds,
              },
            }
          default:
            noCase(value)
        }
      }),
    },
    sampling: block(() => {
      switch (planParamsNorm.advanced.sampling.type) {
        case 'historical':
          return { $case: 'historical', historical: {} }
        case 'monteCarlo':
          return {
            $case: 'monteCarlo',
            monteCarlo: {
              numRuns,
              seed,
              blockSize:
                planParamsNorm.advanced.sampling.data.blockSize.inMonths,
              staggerRunStarts:
                planParamsNorm.advanced.sampling.data.staggerRunStarts,
            },
          }
        default:
          noCase(planParamsNorm.advanced.sampling)
      }
    }),
    annualInflation: block(() => {
      switch (planParamsNorm.advanced.annualInflation.type) {
        case 'suggested':
          return { $case: 'suggested', suggested: {} }
        case 'manual':
          return {
            $case: 'manual',
            manual: planParamsNorm.advanced.annualInflation.value,
          }
        default:
          noCase(planParamsNorm.advanced.annualInflation)
      }
    }),
    strategy: block(() => {
      switch (planParamsNorm.advanced.strategy) {
        case 'TPAW':
          return WirePlanParamsServerStrategy.StrategyTpaw
        case 'SPAW':
          return WirePlanParamsServerStrategy.StrategySpaw
        case 'SWR':
          return WirePlanParamsServerStrategy.StrategySwr
        default:
          noCase(planParamsNorm.advanced.strategy)
      }
    }),
  }

  return {
    evaluationTimestampMs,
    constants,
    ages,
    wealth,
    adjustmentsToSpending,
    risk,
    advanced,
  }
}

getPlanParamsServer.getValidMonthRangeForAmountTimed = (
  amountAndTiming: Exclude<
    NormalizedLabeledAmountTimed['amountAndTiming'],
    { type: 'inThePast' }
  >,
) => {
  switch (amountAndTiming.type) {
    case 'oneTime':
      return amountAndTiming.month.validRangeAsMFN.excludingLocalConstraints
    case 'recurring':
      return amountAndTiming.monthRange.validRangeAsMFN
    default:
      noCase(amountAndTiming)
  }
}
