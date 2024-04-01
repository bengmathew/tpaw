import {
  DialogPosition,
  GlidePathLocation,
  LabeledAmountUntimed,
  LabeledAmountUntimedLocation,
  PLAN_PARAMS_CONSTANTS,
  PlanParams,
  LabeledAmountTimedLocation,
  LabeledAmountTimedList,
  assert,
  fGet,
  noCase,
  LabeledAmountTimedOrUntimedLocation,
} from '@tpaw/common'
import _ from 'lodash'

export namespace PlanParamsHelperFns {
  export const getNextDialogPosition = (
    x: Exclude<DialogPosition, 'done'>,
  ): DialogPosition =>
    fGet(
      PLAN_PARAMS_CONSTANTS.dialogPositionOrder[
        PLAN_PARAMS_CONSTANTS.dialogPositionOrder.indexOf(x) + 1
      ],
    )
  // Intentionally, future savings is allowed even if the couple is jointly
  // retired (withdrawals have started), but one of them are not individually
  // retired. This does not matter either way because future savings and income
  // during retirement are really the same thing under the hood.
  export const getIsFutureSavingsAllowed = (
    isPerson1Retired: boolean,
    isPerson2Retired: boolean | undefined,
  ) =>
    isPerson2Retired === undefined
      ? !isPerson1Retired
      : !(isPerson1Retired && isPerson2Retired)

  type _LabeledAmountUntimedList<LabeledAmountUntimed> =
    | LabeledAmountUntimed[]
    | Record<string, LabeledAmountUntimed>

  type _PlanParamsLabeledAmountUntimedParts<LabeledAmountUntimedList> = {
    adjustmentsToSpending: {
      tpawAndSPAW: {
        legacy: {
          external: LabeledAmountUntimedList
        }
      }
    }
  }

  type _LabeledAmountTimedList<LabeledAmountTimed> =
    | LabeledAmountTimed[]
    | Record<string, LabeledAmountTimed>

  type _PlanParamsLabeledAmountTimedListParts<LabeledAmountTimedList> = {
    wealth: {
      futureSavings: LabeledAmountTimedList
      incomeDuringRetirement: LabeledAmountTimedList
    }
    adjustmentsToSpending: {
      extraSpending: {
        essential: LabeledAmountTimedList
        discretionary: LabeledAmountTimedList
      }
    }
  }

  export const getLabeledAmountUntimedListFromLocation = <
    LabeledAmountUntimedList,
  >(
    planParams: _PlanParamsLabeledAmountUntimedParts<LabeledAmountUntimedList>,
    location: LabeledAmountUntimedLocation,
  ): LabeledAmountUntimedList => {
    switch (location) {
      case 'legacyExternalSources':
        return planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external
      default:
        noCase(location)
    }
  }

  export const getLabeledAmountTimedListFromLocation = <LabeledAmountTimedList>(
    planParams: _PlanParamsLabeledAmountTimedListParts<LabeledAmountTimedList>,
    location: LabeledAmountTimedLocation,
  ): LabeledAmountTimedList => {
    switch (location) {
      case 'futureSavings':
        return planParams.wealth.futureSavings
      case 'incomeDuringRetirement':
        return planParams.wealth.incomeDuringRetirement
      case 'extraSpendingDiscretionary':
        return planParams.adjustmentsToSpending.extraSpending.discretionary
      case 'extraSpendingEssential':
        return planParams.adjustmentsToSpending.extraSpending.essential
      default:
        noCase(location)
    }
  }

  export const getLabeledAmountTimedOrUntimedListFromLocation = <
    LabeledAmountUntimedList,
    LabeledAmountTimedList,
  >(
    planParams: _PlanParamsLabeledAmountUntimedParts<LabeledAmountUntimedList> &
      _PlanParamsLabeledAmountTimedListParts<LabeledAmountTimedList>,
    location: LabeledAmountTimedOrUntimedLocation,
  ): LabeledAmountUntimedList | LabeledAmountTimedList => {
    switch (location) {
      case 'legacyExternalSources':
        return getLabeledAmountUntimedListFromLocation(planParams, location)
      default:
        return getLabeledAmountTimedListFromLocation(planParams, location)
    }
  }

  export const getLabeledAmountTimedListFromLocationAndId = <
    T extends { id: string },
  >(
    planParams: _PlanParamsLabeledAmountTimedListParts<
      _LabeledAmountTimedList<T>
    >,
    location: LabeledAmountTimedLocation,
    id: string,
  ): T | undefined => {
    const amountForMonthRanges = getLabeledAmountTimedListFromLocation(
      planParams,
      location,
    )
    return _.isArray(amountForMonthRanges)
      ? amountForMonthRanges.find((entry) => entry.id === id)
      : amountForMonthRanges[id]
  }

  export const mapAllLabeledAmountTimedList = <LabeledAmountTimed, R = void>(
    planParams: _PlanParamsLabeledAmountTimedListParts<
      _LabeledAmountTimedList<LabeledAmountTimed>
    >,
    fn: (x: LabeledAmountTimed, location: LabeledAmountTimedLocation) => R,
  ): R[] =>
    _.flatten(
      PLAN_PARAMS_CONSTANTS.labeledAmountTimedLocations.map((location) => {
        const amountForMonthRanges = getLabeledAmountTimedListFromLocation(
          planParams,
          location,
        )
        return (
          _.isArray(amountForMonthRanges)
            ? amountForMonthRanges
            : _.values(amountForMonthRanges)
        ).map((x) => fn(x, location))
      }),
    )

  type _PlanParamsForGlidePath<GlidePath> = {
    risk: {
      spawAndSWR: {
        allocation: GlidePath
      }
    }
  }
  export const getGlidePathFromLocation = <GlidePath>(
    planParams: _PlanParamsForGlidePath<GlidePath>,
    location: GlidePathLocation,
  ) => {
    switch (location) {
      case 'spawAndSWRStockAllocation':
        return planParams.risk.spawAndSWR.allocation
      default:
        noCase(location)
    }
  }
  export const forAllGlidePaths = <GlidePath>(
    planParams: _PlanParamsForGlidePath<GlidePath>,
    fn: (x: GlidePath, location: GlidePathLocation) => void,
  ) => {
    PLAN_PARAMS_CONSTANTS.glidePathLocations.forEach((location) => {
      fn(getGlidePathFromLocation(planParams, location), location)
    })
  }

  export const getPerson = (
    params: PlanParams,
    personType: 'person1' | 'person2',
  ) => {
    if (personType === 'person1') return params.people.person1
    assert(params.people.withPartner)
    return params.people.person2
  }
}
