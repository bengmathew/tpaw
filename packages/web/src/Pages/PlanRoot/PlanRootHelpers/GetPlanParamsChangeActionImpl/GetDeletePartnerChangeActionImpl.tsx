import {
  GlidePath,
  GlidePathLocation,
  Month,
  MonthLocation,
  MonthRange,
  PLAN_PARAMS_CONSTANTS,
  PlanParams,
  LabeledAmountTimed,
  LabeledAmountTimedLocation,
  LabeledAmountTimedList,
  assert,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { PlanParamsHelperFns } from '../../../../UseSimulator/PlanParamsHelperFns'
import { PlanParamsChangeActionImpl } from './GetPlanParamsChangeActionImpl'

export const getDeletePartnerChangeActionImpl =
  (): PlanParamsChangeActionImpl => {
    return {
      applyToClone: (clone) => {
        assert(clone.people.withPartner)
        _removePartnerReferences(clone)
        clone.people = {
          withPartner: false,
          person1: clone.people.person1,
        }
      },
      render: () => `Deleted partner`,
      merge: false,
    }
  }

const _doesMonthReferencePartner = (month: Month | { type: 'inThePast' }) => {
  switch (month.type) {
    case 'inThePast':
    case 'calendarMonth':
    case 'calendarMonthAsNow':
      return false
    case 'numericAge':
    case 'namedAge':
      return month.person === 'person2'
    default:
      noCase(month)
  }
}

const _removePartnerReferencesFromLabeledAmountTimedList = (clone: PlanParams) => {
  // Returns a new object only if monthRange is changed.
  const doesMonthRangeReferencePartner = (monthRange: MonthRange) => {
    switch (monthRange.type) {
      case 'startAndEnd': {
        return (
          _doesMonthReferencePartner(monthRange.start) ||
          _doesMonthReferencePartner(monthRange.end)
        )
      }
      case 'startAndDuration': {
        return _doesMonthReferencePartner(monthRange.start)
      }
      case 'endAndDuration': {
        return _doesMonthReferencePartner(monthRange.end)
      }
      default:
        noCase(monthRange)
    }
  }

  // Returns a new object only if amountAndTiming is changed.
  const doesAmountAndTimingReferencePartner = (
    amountAndTiming: LabeledAmountTimed['amountAndTiming'],
  ): boolean => {
    switch (amountAndTiming.type) {
      case 'oneTime':
        return _doesMonthReferencePartner(amountAndTiming.month)
      case 'recurring':
        return doesMonthRangeReferencePartner(amountAndTiming.monthRange)
      case 'inThePast':
        return false
      default:
        noCase(amountAndTiming)
    }
  }

  const removed = new Map<LabeledAmountTimedLocation, LabeledAmountTimed[]>()
  const addToRemoved = (
    location: LabeledAmountTimedLocation,
    value: LabeledAmountTimed,
  ) => {
    const curr = removed.get(location)
    curr ? curr.push(value) : removed.set(location, [value])
  }

  PLAN_PARAMS_CONSTANTS.labeledAmountTimedLocations.forEach((location) => {
    const handleLocation = (
      amountForMonthRanges: LabeledAmountTimedList,
    ): LabeledAmountTimedList => {
      const result: LabeledAmountTimedList = {}
      _.values(amountForMonthRanges).forEach((amountForMonthRange) => {
        if (
          doesAmountAndTimingReferencePartner(amountForMonthRange.amountAndTiming)
        ) {
          addToRemoved(location, amountForMonthRange)
          return
        } else {
          result[amountForMonthRange.id] = amountForMonthRange
        }
      })
      return result
    }
    switch (location) {
      case 'futureSavings':
        clone.wealth.futureSavings = handleLocation(clone.wealth.futureSavings)
        break
      case 'incomeDuringRetirement':
        clone.wealth.incomeDuringRetirement = handleLocation(
          clone.wealth.incomeDuringRetirement,
        )
        break
      case 'extraSpendingEssential':
        clone.adjustmentsToSpending.extraSpending.essential = handleLocation(
          clone.adjustmentsToSpending.extraSpending.essential,
        )
        break
      case 'extraSpendingDiscretionary':
        clone.adjustmentsToSpending.extraSpending.discretionary =
          handleLocation(
            clone.adjustmentsToSpending.extraSpending.discretionary,
          )
        break
      default:
        noCase(location)
    }
  })
  return removed
}

const _removePartnerReferencesFromGlidePaths = (clone: PlanParams) => {
  const glidePathsAdjusted = new Set<GlidePathLocation>()

  PlanParamsHelperFns.forAllGlidePaths(
    clone,
    (glidePath: GlidePath, location) => {
      const intermediateArr = _.values(glidePath.intermediate)
      const newIntermediateArr = intermediateArr.filter(
        (x) => !_doesMonthReferencePartner(x.month),
      )
      if (intermediateArr.length === newIntermediateArr.length) return
      glidePath.intermediate = _.fromPairs(
        newIntermediateArr.map((x) => [x.id, x]),
      )
      glidePathsAdjusted.add(location)
    },
  )
  assert(PLAN_PARAMS_CONSTANTS.glidePathLocations.length === 1)
  assert(
    PLAN_PARAMS_CONSTANTS.glidePathLocations[0] === 'spawAndSWRStockAllocation',
  )
  return {
    spawAndSWRStockAllocationAdjusted: glidePathsAdjusted.has(
      'spawAndSWRStockAllocation',
    ),
  }
}

const _removePartnerReferences = (clone: PlanParams) => {
  // To make sure typechecking fails when new locations are added, so
  // we can update this code.
  const __ = (x: MonthLocation) => {
    switch (x) {
      case 'futureSavings':
      case 'incomeDuringRetirement':
      case 'extraSpendingEssential':
      case 'extraSpendingDiscretionary':
      case 'spawAndSWRStockAllocation':
        return
      default:
        noCase(x)
    }
  }
  _removePartnerReferencesFromLabeledAmountTimedList(clone)
  _removePartnerReferencesFromGlidePaths(clone)
}

export type RemovePartnerAdjustments = Exclude<
  ReturnType<typeof getRemovePartnerAdjustments>,
  null
>
export const getRemovePartnerAdjustments = (
  planParamsNorm: PlanParamsNormalized,
) => {
  const isFutureSavingsGoingToBeAllowed =
    PlanParamsHelperFns.getIsFutureSavingsAllowed(
      planParamsNorm.ages.person1.retirement.isRetired,
      undefined,
    )

  // Note: This is not done during the action, but by normalization once
  // action is applied. We still need to let the user know about them.
  const futureSavingsEntriesToBeRemovedDueSectionRemoval = block(() => {
    const entries = planParamsNorm.wealth.futureSavings
    if (entries.length === 0) return []
    if (isFutureSavingsGoingToBeAllowed) return []
    return entries
  })

  const clone = normalizePlanParamsInverse(planParamsNorm)

  const amountForMonthRangeEntriesToBeRemoved = block(() => {
    const inClone = _removePartnerReferencesFromLabeledAmountTimedList(clone)
    // If future savings is not going to be allowed, these will be handled
    // by futureSavingsEntryRemovals.
    if (!isFutureSavingsGoingToBeAllowed) inClone.delete('futureSavings')
    return new Map(
      [...inClone.entries()].map(([location, entries]) => [
        location,
        entries.map((x) =>
          fGet(
            PlanParamsHelperFns.getLabeledAmountTimedListFromLocationAndId(
              planParamsNorm,
              location,
              x.id,
            ),
          ),
        ),
      ]),
    )
  })

  const { spawAndSWRStockAllocationAdjusted } =
    _removePartnerReferencesFromGlidePaths(clone)

  return futureSavingsEntriesToBeRemovedDueSectionRemoval.length === 0 &&
    amountForMonthRangeEntriesToBeRemoved.size === 0 &&
    !spawAndSWRStockAllocationAdjusted
    ? null
    : {
        futureSavingsEntriesToBeRemovedDueSectionRemoval,
        amountForMonthRangeEntriesToBeRemoved,
        spawAndSWRStockAllocationAdjusted,
      }
}
