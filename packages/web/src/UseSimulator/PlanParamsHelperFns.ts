import {
  CalendarDay,
  CalendarMonthFns,
  DialogPosition,
  GlidePath,
  GlidePathLocation,
  LabeledAmountTimed,
  LabeledAmountTimedList,
  LabeledAmountTimedLocation,
  LabeledAmountTimedOrUntimedLocation,
  LabeledAmountUntimedLocation,
  Month,
  MonthRange,
  PLAN_PARAMS_CONSTANTS,
  Person,
  PersonId,
  PickType,
  PlanParams,
  assert,
  assertFalse,
  block,
  fGet,
  getLastMarketDataDayForUndatedPlans,
  getNYZonedTime,
  getZonedTimeFns,
  noCase,
  planParamsGuard,
} from '@tpaw/common'
import _ from 'lodash'
import { InMonthsFns } from '../Utils/InMonthsFns'
import { PlanParamsNormalized } from './NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from './NormalizePlanParams/NormalizePlanParamsInverse'
import { getFromMFNToNumericAge } from './NormalizePlanParams/NormalizeAges'
import { SimulationInfo } from '../Pages/PlanRoot/PlanRootHelpers/WithSimulation'
import { CurrentPortfolioBalance } from '../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'

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

  export const switchDating = (
    planParamsNorm: PlanParamsNormalized,
    currentPortfolioBalanceInfo: SimulationInfo['currentPortfolioBalanceInfo'],
    ianaTimezoneName: string,
  ) => {
    if (planParamsNorm.datingInfo.isDated) {
      assert(currentPortfolioBalanceInfo.isDatedPlan)
      return toDatelessPlan(
        planParamsNorm,
        CurrentPortfolioBalance.getAmountInfo(currentPortfolioBalanceInfo.info)
          .amount,
      )
    } else {
      return toDatedPlan(planParamsNorm, ianaTimezoneName)
    }
  }

  export const toDatedPlan = (
    planParamsNorm: PlanParamsNormalized,
    ianaTimezoneName: string,
  ) => {
    const { datingInfo } = planParamsNorm
    assert(!datingInfo.isDated)
    const {} = datingInfo
    const timestamp = datingInfo.nowAsTimestampNominal
    const nowAsCalendarMonth = CalendarMonthFns.fromTimestamp(
      datingInfo.nowAsTimestampNominal,
      ianaTimezoneName,
    )
    const _forNowMonth = (): PickType<Month, 'now'> => ({
      type: 'now',
      monthOfEntry: { isDatedPlan: true, calendarMonth: nowAsCalendarMonth },
    })

    const _forMonth = (month: Month): Month => {
      switch (month.type) {
        case 'now':
          return _forNowMonth()
        case 'calendarMonth':
          assertFalse()
        case 'namedAge':
        case 'numericAge':
          return month
        default:
          noCase(month)
      }
    }

    const planParams = mapMonths(
      normalizePlanParamsInverse(planParamsNorm, 'hard'),
      _forMonth,
      _forNowMonth,
    )

    const result: PlanParams = {
      ...planParams,
      timestamp,
      datingInfo: { isDated: true },
      people: block(() => {
        const _forCurrentAgeInfo = (
          x: Person['ages']['currentAgeInfo'],
        ): Person['ages']['currentAgeInfo'] => {
          assert(!x.isDatedPlan)
          return {
            isDatedPlan: true,
            monthOfBirth: CalendarMonthFns.getFromMFN(nowAsCalendarMonth)(
              -x.currentAge.inMonths,
            ),
          }
        }
        const _forPerson = (person: Person): Person => {
          return {
            ages: {
              ...person.ages,
              currentAgeInfo: _forCurrentAgeInfo(person.ages.currentAgeInfo),
            },
          }
        }
        return planParams.people.withPartner
          ? {
              withPartner: planParams.people.withPartner,
              person1: _forPerson(planParams.people.person1),
              person2: _forPerson(planParams.people.person2),
              withdrawalStart: planParams.people.withdrawalStart,
            }
          : {
              withPartner: planParams.people.withPartner,
              person1: _forPerson(planParams.people.person1),
            }
      }),
      wealth: {
        ...planParams.wealth,
        portfolioBalance: {
          isDatedPlan: true,
          updatedHere: true,
          amount: block(() => {
            assert(!planParams.wealth.portfolioBalance.isDatedPlan)
            return planParams.wealth.portfolioBalance.amount
          }),
        },
      },
      results: null,
    }

    const check = planParamsGuard(result)
    if (check.error) {
      console.dir(result)
      console.dir(check.message)
      assertFalse()
    }
    return result
  }

  export const toDatelessPlan = (
    planParamsNorm: PlanParamsNormalized,
    currentPortfolioBalance: number,
  ): PlanParams => {
    const { datingInfo } = planParamsNorm
    assert(datingInfo.isDated)
    const { nowAsTimestamp: timestamp, nowAsCalendarDay } = datingInfo
    const calendarMonthToMFN = CalendarMonthFns.getToMFN(nowAsCalendarDay)
    const mfnToNumericAge = getFromMFNToNumericAge(planParamsNorm)

    const _forNowMonth = (): PickType<Month, 'now'> => ({
      type: 'now',
      monthOfEntry: { isDatedPlan: false },
    })

    const _forMonth = (month: Month): Month => {
      switch (month.type) {
        case 'now':
          return _forNowMonth()
        case 'calendarMonth':
          return mfnToNumericAge.auto(calendarMonthToMFN(month.calendarMonth))
        case 'namedAge':
        case 'numericAge':
          return month
        default:
          noCase(month)
      }
    }

    const planParams = mapMonths(
      normalizePlanParamsInverse(planParamsNorm, 'hard'),
      _forMonth,
      _forNowMonth,
    )

    const result: PlanParams = {
      ...planParams,
      timestamp,
      datingInfo: {
        isDated: false,
        marketDataAsOfEndOfDayInNY:
          getLastMarketDataDayForUndatedPlans(timestamp),
      },
      people: block(() => {
        const _forCurrentAgeInfo = (
          x: Person['ages']['currentAgeInfo'],
        ): Person['ages']['currentAgeInfo'] => {
          assert(x.isDatedPlan)
          return {
            isDatedPlan: false,
            currentAge: {
              inMonths: CalendarMonthFns.diff(nowAsCalendarDay, x.monthOfBirth),
            },
          }
        }
        const _forPerson = (person: Person): Person => {
          return {
            ages: {
              ...person.ages,
              currentAgeInfo: _forCurrentAgeInfo(person.ages.currentAgeInfo),
            },
          }
        }
        return planParams.people.withPartner
          ? {
              withPartner: planParams.people.withPartner,
              person1: _forPerson(planParams.people.person1),
              person2: _forPerson(planParams.people.person2),
              withdrawalStart: planParams.people.withdrawalStart,
            }
          : {
              withPartner: planParams.people.withPartner,
              person1: _forPerson(planParams.people.person1),
            }
      }),
      wealth: {
        ...planParams.wealth,
        portfolioBalance: {
          isDatedPlan: false,
          amount: currentPortfolioBalance,
        },
      },
      results: null,
    }
    assert(!planParamsGuard(result).error)
    return result
  }

  export const mapMonths = (
    planParams: PlanParams,
    fn: (month: Month) => Month,
    glidePathStart: (month: PickType<Month, 'now'>) => PickType<Month, 'now'>,
  ): PlanParams => {
    const _mapMonthRange = (monthRange: MonthRange): MonthRange => {
      switch (monthRange.type) {
        case 'startAndEnd':
          return {
            type: 'startAndEnd',
            start: fn(monthRange.start),
            end:
              monthRange.end.type === 'inThePast'
                ? monthRange.end
                : fn(monthRange.end),
          }
        case 'startAndDuration':
          return {
            type: 'startAndDuration',
            start: fn(monthRange.start),
            duration: monthRange.duration,
          }
        case 'endAndDuration':
          return {
            type: 'endAndDuration',
            end: fn(monthRange.end),
            duration: monthRange.duration,
          }
        default:
          noCase(monthRange)
      }
    }

    const _mapLabeledAmountTimed = (
      x: LabeledAmountTimed,
    ): LabeledAmountTimed => {
      return {
        ...x,
        amountAndTiming: block(() => {
          switch (x.amountAndTiming.type) {
            case 'inThePast':
              return x.amountAndTiming
            case 'oneTime':
              return {
                ...x.amountAndTiming,
                month: fn(x.amountAndTiming.month),
              }
            case 'recurring':
              return {
                ...x.amountAndTiming,
                monthRange: _mapMonthRange(x.amountAndTiming.monthRange),
              }
            default:
              noCase(x.amountAndTiming)
          }
        }),
      }
    }

    const _mapLabeledAmountTimedList = (
      x: LabeledAmountTimedList,
    ): LabeledAmountTimedList => _.mapValues(x, _mapLabeledAmountTimed)

    const _mapGlidePath = (x: GlidePath): GlidePath => ({
      ...x,
      start: { ...x.start, month: glidePathStart(x.start.month) },
      intermediate: _.mapValues(x.intermediate, (x) => ({
        ...x,
        month: fn(x.month),
      })),
    })

    return {
      ...planParams,
      wealth: {
        ...planParams.wealth,
        futureSavings: _mapLabeledAmountTimedList(
          planParams.wealth.futureSavings,
        ),
        incomeDuringRetirement: _mapLabeledAmountTimedList(
          planParams.wealth.incomeDuringRetirement,
        ),
      },
      adjustmentsToSpending: {
        ...planParams.adjustmentsToSpending,
        extraSpending: {
          essential: _mapLabeledAmountTimedList(
            planParams.adjustmentsToSpending.extraSpending.essential,
          ),
          discretionary: _mapLabeledAmountTimedList(
            planParams.adjustmentsToSpending.extraSpending.discretionary,
          ),
        },
      },
      risk: {
        ...planParams.risk,
        spawAndSWR: {
          allocation: _mapGlidePath(planParams.risk.spawAndSWR.allocation),
        },
      },
    }
  }
}
