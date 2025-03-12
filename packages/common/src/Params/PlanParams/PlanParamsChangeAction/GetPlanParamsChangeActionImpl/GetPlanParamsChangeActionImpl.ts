import _ from 'lodash'
import { assert, assertFalse, fGet, noCase, optGet } from '../../../../Utils'
import {
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  partialDefaultDatelessPlanParams,
} from '../../DefaultPlanParams'
import { PlanParamsNormalized } from '../../NormalizePlanParams/NormalizePlanParams'
import {
  LabeledAmountTimed,
  LabeledAmountTimedLocation,
  LabeledAmountUntimed,
  LabeledAmountUntimedLocation,
  PlanParams,
} from '../../PlanParams'
import {
  PlanParamsChangeAction,
  PlanParamsChangeActionCurrent,
} from '../PlanParamsChangeAction'
import { PlanParamsHelperFns } from '../../PlanParamsHelperFns'
import { getDeletePartnerChangeActionImpl } from './GetDeletePartnerChangeActionImpl'
import { getSetPersonRetiredChangeActionImpl } from './GetSetPersonRetiredChangeActionImpl'

const { getPerson } = PlanParamsHelperFns

export type PlanParamsChangeActionImpl = {
  applyToClone: (
    clone: PlanParams,
    planParamsNorm: PlanParamsNormalized,
  ) => void | PlanParams // If a value is returned, it supercedes the clone.
  merge: false | true | ((prev: PlanParamsChangeAction) => boolean)
}

export const getPlanParamsChangeActionImpl = (
  action: PlanParamsChangeActionCurrent,
): PlanParamsChangeActionImpl => {
  switch (action.type) {
    case 'start':
      return {
        applyToClone: () => assertFalse(),
        merge: false,
      }
    case 'startCopiedFromBeforeHistory':
      return {
        applyToClone: () => assertFalse(),
        merge: false,
      }
    case 'startCutByClient':
      return {
        applyToClone: () => assertFalse(),
        merge: false,
      }
    case 'startFromURL':
      return {
        applyToClone: () => assertFalse(),

        merge: false,
      }
    case 'noOpToMarkMigration':
      return {
        applyToClone: () => {}, // Do nothing.
        merge: false,
      }
    // ---------
    // SetMarketDataDay
    // ---------
    case 'setMarketDataDay': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(!clone.datingInfo.isDated)
          clone.datingInfo.marketDataAsOfEndOfDayInNY = value
        },
        merge: false,
      }
    }

    // ---------
    // SetDialogPosition
    // ---------
    case 'setDialogPosition': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.dialogPositionNominal = value
        },
        merge: false,
      }
    }

    // ---------
    // AddPartner
    // ---------
    case 'addPartner': {
      return {
        applyToClone: (clone) => {
          assert(!clone.people.withPartner)
          clone.people = {
            withPartner: true,
            person1: clone.people.person1,
            person2: _.cloneDeep(clone.people.person1),
            withdrawalStart: 'person1',
          }
        },
        merge: false,
      }
    }

    // ---------
    // DeletePartner
    // ---------
    case 'deletePartner': {
      return getDeletePartnerChangeActionImpl()
    }
    // ---------
    // SetPersonRetired
    // ---------
    case 'setPersonRetired':
      return getSetPersonRetiredChangeActionImpl(action)

    // ---------
    // SetPersonNotRetired
    // ---------
    case 'setPersonNotRetired': {
      const personType = action.value
      return {
        applyToClone: (clone, planParamsNorm) => {
          const currPerson = getPerson(clone, personType)
          const defaultPerson = _.cloneDeep(
            partialDefaultDatelessPlanParams.people.person1,
          )
          assert(defaultPerson.ages.type === 'retirementDateSpecified')
          const retirementAge = {
            inMonths: _.clamp(
              defaultPerson.ages.retirementAge.inMonths,
              fGet(planParamsNorm.ages[personType]).currentAgeInfo.inMonths + 1,
              currPerson.ages.maxAge.inMonths - 1,
            ),
          }
          currPerson.ages = {
            type: 'retirementDateSpecified',
            currentAgeInfo: currPerson.ages.currentAgeInfo,
            maxAge: currPerson.ages.maxAge,
            retirementAge,
          }
        },
        merge: false,
      }
    }
    // ---------
    // setPersonCurrentAgeInfo
    // ---------
    case 'setPersonCurrentAgeInfo': {
      const { personId, currentAgeInfo } = action.value
      return {
        applyToClone: (clone) => {
          getPerson(clone, personId).ages.currentAgeInfo = currentAgeInfo
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.personId === personId,
      }
    }

    // ---------
    // SetPersonRetirementAge
    // ---------
    case 'setPersonRetirementAge': {
      const { person: personType, retirementAge } = action.value
      return {
        applyToClone: (clone) => {
          const person = getPerson(clone, personType)
          assert(person.ages.type === 'retirementDateSpecified')
          person.ages.retirementAge = retirementAge
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.person === personType,
      }
    }

    // ---------
    // SetPersonMaxAge
    // ---------
    case 'setPersonMaxAge': {
      const { person: personType, maxAge } = action.value
      return {
        applyToClone: (clone) => {
          const person = getPerson(clone, personType)
          person.ages.maxAge = maxAge
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.person === personType,
      }
    }

    // ---------
    // SetWithdrawalStart
    // ---------
    case 'setWithdrawalStart': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.people.withPartner)
          clone.people.withdrawalStart = value
        },
        merge: false,
      }
    }

    // -------------- LABELED AMOUNT TIMED/UNTIMED -------------------
    // ---------
    // addLabeledAmountUntimed
    // ---------
    case 'addLabeledAmountUntimed': {
      const { location, entryId, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          const entries =
            PlanParamsHelperFns.getLabeledAmountUntimedListFromLocation(
              clone,
              location,
            )
          assert(!optGet(entries, entryId))
          entries[entryId] = {
            id: entryId,
            label: null,
            amount: 0,
            nominal: false,
            sortIndex,
            colorIndex: _getNextColorIndex(
              _getUsedColorIndexesByLocation(clone, location),
            ),
          }
        },
        merge: false,
      }
    }
    // ---------
    // addLabeledAmountTimed2
    // ---------
    case 'addLabeledAmountTimed2': {
      const { location, entryId, amountAndTiming, sortIndex } = action.value
      return {
        applyToClone: (clone) => {
          const values =
            PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
              clone,
              location,
            )
          assert(!optGet(values, entryId))

          values[entryId] = {
            id: entryId,
            sortIndex,
            label: null,
            nominal: false,
            colorIndex: _getNextColorIndex(
              _getUsedColorIndexesByLocation(clone, location),
            ),
            amountAndTiming,
          }
        },
        merge: false,
      }
    }

    // ---------
    // DeleteLabeledAmount
    // ---------
    case 'deleteLabeledAmountTimedOrUntimed': {
      const { location, entryId } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          assert(optGet(entries, entryId))
          delete entries[entryId]
        },
        merge: false,
      }
    }
    // ---------
    // setLabelForLabeledAmountTimedOrUntimed
    // ---------
    case 'setLabelForLabeledAmountTimedOrUntimed': {
      const { location, entryId, label } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          fGet(optGet(entries, entryId)).label = label
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.entryId === entryId,
      }
    }

    // ---------
    // setAmountForLabeledAmountUntimed
    // ---------
    case 'setAmountForLabeledAmountUntimed': {
      const { location, entryId, amount } = action.value
      return {
        applyToClone: (clone) => {
          fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountUntimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          ).amount = amount
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.entryId === entryId,
      }
    }

    // ---------
    // setBaseAmountForLabeledAmountTimed
    // ---------
    case 'setBaseAmountForLabeledAmountTimed': {
      const { location, entryId, baseAmount } = action.value
      return {
        applyToClone: (clone) => {
          const entry = fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          )
          assert(entry.amountAndTiming.type === 'recurring')
          entry.amountAndTiming.baseAmount = baseAmount
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.entryId === entryId,
      }
    }

    // ---------
    // setNominalForLabeledAmountTimedOrUntimed
    // ---------
    case 'setNominalForLabeledAmountTimedOrUntimed': {
      const { location, entryId, nominal } = action.value
      return {
        applyToClone: (clone) => {
          const entries: Record<
            string,
            LabeledAmountTimed | LabeledAmountUntimed
          > =
            PlanParamsHelperFns.getLabeledAmountTimedOrUntimedListFromLocation(
              clone,
              location,
            )
          fGet(optGet(entries, entryId)).nominal = nominal
        },
        merge: false,
      }
    }
    // ---------
    // setMonthRangeForLabeledAmountTimed2
    // ---------
    case 'setMonthRangeForLabeledAmountTimed2': {
      const { location, entryId, monthRange } = action.value
      return {
        applyToClone: (clone) => {
          const entry = fGet(
            optGet(
              PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
                clone,
                location,
              ),
              entryId,
            ),
          )
          assert(entry.amountAndTiming.type !== 'oneTime')
          assert(entry.amountAndTiming.type !== 'inThePast')
          entry.amountAndTiming.monthRange = monthRange
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.entryId === entryId,
      }
    }

    // --------------------------- WEALTH --------------------------------
    // ---------
    // SetCurrentPortfolioBalance
    // ---------
    case 'setCurrentPortfolioBalance': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.wealth.portfolioBalance = clone.datingInfo.isDated
            ? {
                isDatedPlan: true,
                updatedHere: true,
                amount: value,
              }
            : {
                isDatedPlan: false,
                amount: value,
              }
        },
        merge: () => true,
      }
    }
    // ---------
    // SetSpendingCeiling
    // ---------
    case 'setSpendingCeiling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling = value
        },
        merge: (prev) =>
          prev.type === action.type && prev.value !== null && value !== null,
      }
    }
    // ---------
    // SetSpendingFloor
    // ---------
    case 'setSpendingFloor': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor = value
        },
        merge: (prev) =>
          prev.type === action.type && prev.value !== null && value !== null,
      }
    }

    // ---------
    // SetLegacyTotal
    // ---------
    case 'setLegacyTotal': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.adjustmentsToSpending.tpawAndSPAW.legacy.total = value
        },
        merge: () => true,
      }
    }
    // --------------------------- ADVANCED --------------------------------
    // ---------
    // SetTPAWRiskTolerance
    // ---------
    case 'setTPAWRiskTolerance': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.at20 = value
        },
        merge: () => true,
      }
    }
    // ---------
    // SetTPAWAdditionalSpendingTilt
    // ---------
    case 'setTPAWAdditionalSpendingTilt': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.additionalAnnualSpendingTilt = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWRiskDeltaAtMaxAge
    // ---------
    case 'setTPAWRiskDeltaAtMaxAge': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.deltaAtMaxAge = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWRiskToleranceForLegacyAsDeltaFromAt20
    // ---------
    case 'setTPAWRiskToleranceForLegacyAsDeltaFromAt20': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWTimePreference
    // ---------
    case 'setTPAWTimePreference': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpaw.timePreference = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetSWRWithdrawalAsPercentPerYear
    // ---------
    case 'setSWRWithdrawalAsPercentPerYear': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.swr.withdrawal = {
            type: 'asPercentPerYear',
            percentPerYear: value,
          }
        },
        merge: () => true,
      }
    }

    // ---------
    // SetSWRWithdrawalAsAmountPerMonth
    // ---------
    case 'setSWRWithdrawalAsAmountPerMonth': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.swr.withdrawal = {
            type: 'asAmountPerMonth',
            amountPerMonth: value,
          }
        },
        merge: () => true,
      }
    }

    // ---------
    // setSPAWAndSWRAllocation2
    // ---------
    case 'setSPAWAndSWRAllocation2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.spawAndSWR.allocation = value
        },
        merge: () => true,
      }
    }
    // ---------
    // SetSPAWAnnualSpendingTilt
    // ---------
    case 'setSPAWAnnualSpendingTilt': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.spaw.annualSpendingTilt = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetTPAWAndSPAWLMP
    // ---------
    case 'setTPAWAndSPAWLMP': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.risk.tpawAndSPAW.lmp = value
        },

        merge: () => true,
      }
    }

    // ---------
    // setExpectedReturnsForPlanning
    // ---------
    case 'setExpectedReturnsForPlanning': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.returnsStatsForPlanning.expectedValue.empiricalAnnualNonLog =
            value
        },
        merge: (prev) =>
          prev.type === action.type && prev.value.type === value.type,
      }
    }

    // ---------
    // setReturnsStatsForPlanningStockVolatilityScale
    // ---------
    case 'setReturnsStatsForPlanningStockVolatilityScale': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.returnsStatsForPlanning.standardDeviation.stocks.scale.log =
            value
        },
        merge: true,
      }
    }

    // ---------
    // setHistoricalReturnsAdjustmentBondVolatilityScale
    // ---------
    case 'setHistoricalReturnsAdjustmentBondVolatilityScale': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalReturnsAdjustment.standardDeviation.bonds.scale.log =
            value
        },
        merge: true,
      }
    }

    // ---------
    // SetAnnualInflation
    // ---------
    case 'setAnnualInflation': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.annualInflation = value
        },
        merge: (prev) =>
          prev.type === action.type &&
          prev.value.type === 'manual' &&
          value.type === 'manual',
      }
    }

    // ---------
    // SetSamplingToDefault
    // ---------
    case 'setSamplingToDefault': {
      return {
        applyToClone: (clone) => {
          const defaultSampling = _.clone(
            partialDefaultDatelessPlanParams.advanced.sampling,
          )
          assert(defaultSampling.type === 'monteCarlo')
          // Recreating this object instead of using defaultSampling makes sure we
          // are adding default data correctly. There is none currently , but if
          // gets added it will lead to a compilation error here.
          clone.advanced.sampling = {
            type: defaultSampling.type,
            data: {
              blockSize: { inMonths: defaultSampling.data.blockSize.inMonths },
              staggerRunStarts: defaultSampling.data.staggerRunStarts,
            },
          }
        },
        merge: false,
      }
    }

    // ---------
    // SetSampling
    // ---------
    case 'setSampling': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          switch (value) {
            case 'monteCarlo':
              if (clone.advanced.sampling.type === 'monteCarlo') return
              const defaultSampling = _.cloneDeep(
                partialDefaultDatelessPlanParams.advanced.sampling,
              )
              assert(defaultSampling.type === 'monteCarlo')
              clone.advanced.sampling = {
                type: 'monteCarlo',
                data:
                  clone.advanced.sampling.defaultData.monteCarlo ??
                  defaultSampling.data,
              }
              break
            case 'historical':
              if (clone.advanced.sampling.type === 'historical') return
              clone.advanced.sampling = {
                type: 'historical',
                defaultData: { monteCarlo: clone.advanced.sampling.data },
              }
              break
            default:
              noCase(value)
          }
        },
        merge: false,
      }
    }

    // ---------
    // SetMonteCarloSamplingBlockSize2
    // ---------
    case 'setMonteCarloSamplingBlockSize2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.sampling.type === 'monteCarlo')
          clone.advanced.sampling.data.blockSize = value
        },
        merge: () => true,
      }
    }

    // ---------
    // SetMonteCarloSamplingStaggerRunStarts
    // ---------
    case 'setMonteCarloStaggerRunStarts': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          assert(clone.advanced.sampling.type === 'monteCarlo')
          clone.advanced.sampling.data.staggerRunStarts = value
        },
        merge: false,
      }
    }

    // ---------
    // SetStrategy
    // ---------
    case 'setStrategy': {
      const { value } = action
      return {
        applyToClone: (clone, planParamsNorm) => {
          clone.advanced.strategy = value
          if (value === 'SWR' && clone.risk.swr.withdrawal.type === 'default') {
            clone.risk.swr.withdrawal = {
              type: 'asPercentPerYear',
              percentPerYear: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                planParamsNorm.ages.simulationMonths.numWithdrawalMonths,
              ),
            }
          }
        },
        merge: false,
      }
    }

    // ---------
    // setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2
    // ---------
    case 'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2': {
      const { value } = action
      return {
        applyToClone: (clone) => {
          clone.advanced.historicalReturnsAdjustment.overrideToFixedForTesting =
            value
        },
        merge: false,
      }
    }
  }
}

const _getUsedColorIndexesByLocation = (
  planParams: PlanParams,
  location: LabeledAmountUntimedLocation | LabeledAmountTimedLocation,
): Set<number> => {
  switch (location) {
    case 'extraSpendingEssential':
    case 'extraSpendingDiscretionary':
      return new Set(
        [
          ..._.values(planParams.adjustmentsToSpending.extraSpending.essential),
          ..._.values(
            planParams.adjustmentsToSpending.extraSpending.discretionary,
          ),
        ].map((x) => x.colorIndex),
      )
    case 'futureSavings':
    case 'incomeDuringRetirement':
      return new Set(
        [
          ..._.values(planParams.wealth.futureSavings),
          ..._.values(planParams.wealth.incomeDuringRetirement),
        ].map((x) => x.colorIndex),
      )
    case 'legacyExternalSources':
      return new Set(
        [
          ..._.values(
            planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external,
          ),
        ].map((x) => x.colorIndex),
      )
    default:
      noCase(location)
  }
}

const _getNextColorIndex = (usedColorIndexes: Set<number>) => {
  let colorIndex = 0
  while (true) {
    if (!usedColorIndexes.has(colorIndex)) return colorIndex
    colorIndex++
  }
}
