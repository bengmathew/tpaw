import {
  CalendarMonth,
  GlidePath,
  Month,
  MonthRange,
  Person,
  PlanParams,
  ValueForMonthRange,
  ValueForMonthRanges,
  assert,
  block,
  fGet,
  letIn,
  noCase,
  planParamsFns,
} from '@tpaw/common'
import { Record } from '../Utils/Record'
import { PlanParamsExtended } from './ExtentPlanParams'
import _ from 'lodash'
import jsonpatch from 'fast-json-patch'

// TODO: Replace as many planParamsExt uses with planParamsNorm as possible.
export type PlanParamsNormalized = ReturnType<typeof normalizePlanParams>
export const normalizePlanParams = (planParamsExt: PlanParamsExtended) => {
  const { planParams, getZonedTime, currentTimestamp } = planParamsExt
  const nowAsCalendarMonth = letIn(
    getZonedTime(currentTimestamp),
    (currentDateTime) => ({
      year: currentDateTime.year,
      month: currentDateTime.month,
    }),
  )

  const result = normalizePlanParamsJS(planParams, nowAsCalendarMonth)
  // TODO
  const usingExt = _getUsingExt(planParamsExt)
  if (!_.isEqual(result, usingExt)) {
    throw new Error(
      `normalizePlanParamsJS and _getUsingExt are not equal\n: ${JSON.stringify(jsonpatch.compare(result, usingExt))}\n${JSON.stringify(jsonpatch.compare(usingExt, result))}`,
    )
  }
  return result
}

export const normalizePlanParamsJS = (
  planParams: PlanParams,
  nowAsCalendarMonth: CalendarMonth,
) => {
  const asMFNForCalendarMonth = _getAsMFNForCalendarMonth(nowAsCalendarMonth)

  const ages = _getAges(planParams.people, asMFNForCalendarMonth)
  const asMFNForMonth = _getAsMFNForMonth(ages, asMFNForCalendarMonth)
  const asMFNForMonthRange = (monthRange: MonthRange) => {
    switch (monthRange.type) {
      case 'startAndEnd':
        return {
          start: asMFNForMonth(monthRange.start),
          end: asMFNForMonth(monthRange.end),
        }
      case 'startAndNumMonths': {
        const start = asMFNForMonth(monthRange.start)
        return {
          start,
          end: start + monthRange.numMonths - 1,
        }
      }
      case 'endAndNumMonths': {
        const end = asMFNForMonth(monthRange.end)
        return {
          start: end - (monthRange.numMonths - 1),
          end,
        }
      }
      default:
        noCase(monthRange)
    }
  }

  const normalizeValueForMonthRange = (orig: ValueForMonthRange) => {
    {
      const { monthRange, ...rest } = orig
      return { ...rest, monthRange: asMFNForMonthRange(orig.monthRange) }
    }
  }
  const normalizeValueForMonthRanges = (orig: ValueForMonthRanges) =>
    Record.mapValues(orig, normalizeValueForMonthRange)

  const normalizeGlidePath = (orig: GlidePath) => {
    return {
      start: {
        month: asMFNForCalendarMonth(orig.start.month),
        stocks: orig.start.stocks,
      },
      intermediate: Record.mapValues(orig.intermediate, (orig) => ({
        id: orig.id,
        month: asMFNForMonth(orig.month),
        indexToSortByAdded: orig.indexToSortByAdded,
        stocks: orig.stocks,
      })),
      end: orig.end,
    }
  }

  const result = {
    v: planParams.v,
    timestamp: planParams.timestamp,
    dialogPositionNominal: planParams.dialogPositionNominal,

    ages,

    wealth: {
      portfolioBalance: planParams.wealth.portfolioBalance,
      futureSavings: normalizeValueForMonthRanges(
        planParams.wealth.futureSavings,
      ),
      incomeDuringRetirement: normalizeValueForMonthRanges(
        planParams.wealth.incomeDuringRetirement,
      ),
    },

    adjustmentsToSpending: {
      extraSpending: {
        essential: normalizeValueForMonthRanges(
          planParams.adjustmentsToSpending.extraSpending.essential,
        ),
        discretionary: normalizeValueForMonthRanges(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
        ),
      },
      tpawAndSPAW: planParams.adjustmentsToSpending.tpawAndSPAW,
    },

    risk: {
      tpaw: planParams.risk.tpaw,
      tpawAndSPAW: planParams.risk.tpawAndSPAW,
      spaw: planParams.risk.spaw,
      spawAndSWR: {
        allocation: normalizeGlidePath(planParams.risk.spawAndSWR.allocation),
      },
      swr: planParams.risk.swr,
    },
    advanced: planParams.advanced,
    results: planParams.results,
  }

  return result
}

export const _getAsMFNForCalendarMonth =
  (nowAsCalendarMonth: CalendarMonth) =>
  ({ year, month }: CalendarMonth) =>
    (year - nowAsCalendarMonth.year) * 12 + month - nowAsCalendarMonth.month

export const _getAges = (
  orig: PlanParams['people'],
  asMFNForCalendarMonth: (x: CalendarMonth) => number,
) => {
  const person1 = _getAgesForPerson(orig.person1.ages, asMFNForCalendarMonth)
  const person2 = orig.withPartner
    ? _getAgesForPerson(orig.person2.ages, asMFNForCalendarMonth)
    : null
  const simulationMonths = _getSimulationMonths(
    person1,
    person2,
    orig.withPartner ? orig.withdrawalStart : 'person1',
  )
  const validMonthRanges = block(() => {
    const futureSavingsAsMFN = _getValidMonthRangeForFutureSavingsAsMFN(
      person1,
      person2,
    )
    assert(
      planParamsFns.getIsFutureSavingsAllowed(
        person1.retirement.isRetired,
        person2?.retirement.isRetired,
      ) ===
        (futureSavingsAsMFN !== null),
    )
    return {
      futureSavingsAsMFN,
      incomeDuringRetirementAsMFN:
        _getValidMonthRangeForIncomeDuringRetirementAsMFN(
          person1,
          person2,
          simulationMonths,
        ),
      extraSpending: _getValidMonthRangeForExtraSpendingAsMFN(simulationMonths),
    }
  })
  return {
    person1,
    person2,
    simulationMonths,
    validMonthRanges,
  }
}

export const _getAsMFNForMonth =
  (
    {
      person1,
      person2,
    }: {
      person1: ReturnType<typeof _getAgesForPerson>
      person2: ReturnType<typeof _getAgesForPerson> | null
    },
    asMFNForCalendarMonth: (x: CalendarMonth) => number,
  ) =>
  (month: Month) => {
    switch (month.type) {
      case 'calendarMonthAsNow':
        return asMFNForCalendarMonth(month.monthOfEntry)
      case 'calendarMonth':
        return asMFNForCalendarMonth(month.calendarMonth)
      case 'namedAge': {
        const person = fGet({ person1, person2 }[month.person])
        switch (month.age) {
          case 'lastWorkingMonth':
            return fGet(person.retirement.ageAsMFNIfSpecifiedElseNull) - 1
          case 'retirement':
            return fGet(person.retirement.ageAsMFNIfSpecifiedElseNull)
          case 'max':
            return person.maxAgeAsMFN
          default:
            noCase(month)
        }
      }
      case 'numericAge': {
        const person = fGet({ person1, person2 }[month.person])
        return person.monthOfBirthAsMFN + month.age.inMonths
      }
      default:
        noCase(month)
    }
  }

export const _getAgesForPerson = (
  orig: Person['ages'],
  asMFNForCalendarMonth: PlanParamsExtended['calendarMonthToMonthsFromNow'],
) => {
  const monthOfBirthAsMFN = asMFNForCalendarMonth(orig.monthOfBirth)
  return {
    monthOfBirthAsMFN,
    maxAgeAsMFN: monthOfBirthAsMFN + orig.maxAge.inMonths,
    retirement: block(() => {
      switch (orig.type) {
        case 'retiredWithNoRetirementDateSpecified':
          return {
            ageAsMFNIfInFutureElseNull: null as number | null,
            ageAsMFNIfSpecifiedElseNull: null as number | null,
            isRetired: true,
          }
        case 'retirementDateSpecified':
          const ageAsMFNIfSpecifiedElseNull =
            monthOfBirthAsMFN + orig.retirementAge.inMonths
          const ageAsMFNIfInFutureElseNull =
            ageAsMFNIfSpecifiedElseNull > 0 ? ageAsMFNIfSpecifiedElseNull : null
          return {
            ageAsMFNIfInFutureElseNull,
            ageAsMFNIfSpecifiedElseNull,
            isRetired: ageAsMFNIfInFutureElseNull === null,
          }
        default:
          noCase(orig)
      }
    }),
  }
}

export const _getSimulationMonths = (
  person1: ReturnType<typeof _getAgesForPerson>,
  person2: ReturnType<typeof _getAgesForPerson> | null,
  withdrawalStartsAtRetirementOf: 'person1' | 'person2',
) => {
  const lastMonthAsMFN = Math.max(
    person1.maxAgeAsMFN,
    person2?.maxAgeAsMFN ?? 0,
  )

  const withdrawalStartMonthAsMFN = block(() => {
    const withdrawalStartPerson = fGet(
      { person1, person2 }[withdrawalStartsAtRetirementOf],
    )
    const result =
      withdrawalStartPerson.retirement.ageAsMFNIfInFutureElseNull ?? 0
    return result
  })
  return {
    numMonths: lastMonthAsMFN + 1,
    lastMonthAsMFN,
    withdrawalStartMonthAsMFN,
  }
}

export const _getValidMonthRangeForFutureSavingsAsMFN = (
  person1: { retirement: { ageAsMFNIfInFutureElseNull: number | null } },
  person2: { retirement: { ageAsMFNIfInFutureElseNull: number | null } } | null,
) => {
  if (person2 === null) {
    return person1.retirement.ageAsMFNIfInFutureElseNull === null
      ? null
      : { start: 0, end: person1.retirement.ageAsMFNIfInFutureElseNull - 1 }
  } else {
    if (
      person1.retirement.ageAsMFNIfInFutureElseNull === null &&
      person2.retirement.ageAsMFNIfInFutureElseNull === null
    )
      return null
    return {
      start: 0,
      end:
        Math.max(
          person1.retirement.ageAsMFNIfInFutureElseNull ?? 0,
          person2.retirement.ageAsMFNIfInFutureElseNull ?? 0,
        ) - 1,
    }
  }
}

export const _getValidMonthRangeForIncomeDuringRetirementAsMFN = (
  person1: { retirement: { ageAsMFNIfInFutureElseNull: number | null } },
  person2: { retirement: { ageAsMFNIfInFutureElseNull: number | null } } | null,
  simulationMonths: { lastMonthAsMFN: number },
) =>
  person2 === null
    ? {
        start: person1.retirement.ageAsMFNIfInFutureElseNull ?? 0,
        end: simulationMonths.lastMonthAsMFN,
      }
    : {
        start: Math.min(
          person1.retirement.ageAsMFNIfInFutureElseNull ?? 0,
          person2.retirement.ageAsMFNIfInFutureElseNull ?? 0,
        ),
        end: simulationMonths.lastMonthAsMFN,
      }

export const _getValidMonthRangeForExtraSpendingAsMFN = (simulationMonths: {
  lastMonthAsMFN: number
}) => ({ start: 0, end: simulationMonths.lastMonthAsMFN })

export type NormalizedGlidePathEntry =
  | {
      type: 'start'
      month: number
      stocks: number
      issue: 'none'
    }
  | {
      type: 'end'
      month: number
      stocks: number
      issue: 'none'
    }
  | {
      type: 'intermediate'
      id: string
      month: number
      stocks: number
      issue: 'none' | 'beforeStart' | 'afterEnd' | 'duplicate'
    }

const _getUsingExt = (
  planParamsExt: PlanParamsExtended,
): PlanParamsNormalized => {
  const {
    planParams,
    asMFN,
    pickPerson,
    months,
    isPersonRetired,
    validMonthRangeAsMFN,
    numMonths,
    withdrawalStartMonth,
    maxMaxAge,
  } = planParamsExt

  const normalizeValueForMonthRange = (orig: ValueForMonthRange) => {
    {
      const { monthRange, ...rest } = orig
      return { ...rest, monthRange: asMFN(orig.monthRange) }
    }
  }
  const normalizeValueForMonthRanges = (orig: ValueForMonthRanges) =>
    Record.mapValues(orig, normalizeValueForMonthRange)
  const normalizeGlidePath = (orig: GlidePath) => {
    return {
      start: {
        month: asMFN(orig.start.month),
        stocks: orig.start.stocks,
      },
      intermediate: Record.mapValues(orig.intermediate, (orig) => ({
        id: orig.id,
        month: asMFN(orig.month),
        indexToSortByAdded: orig.indexToSortByAdded,
        stocks: orig.stocks,
      })),
      end: orig.end,
    }
  }
  const ages = block(() => {
    const _getAgesForP = (
      personType: 'person1' | 'person2',
    ): ReturnType<typeof _getAgesForPerson> => {
      const person = pickPerson(personType)
      const monthOfBirthAsMFN = asMFN(person.ages.monthOfBirth)
      const maxAgeAsMFN = asMFN(months[personType].max)
      return {
        monthOfBirthAsMFN,
        maxAgeAsMFN,
        retirement: block(() => {
          const ageAsMFN = asMFN(months[personType].retirement)
          return {
            ageAsMFNIfInFutureElseNull: ageAsMFN > 0 ? ageAsMFN : null,
            isRetired: isPersonRetired(personType),
            ageAsMFNIfSpecifiedElseNull:
              person.ages.type === 'retiredWithNoRetirementDateSpecified'
                ? null
                : ageAsMFN,
          }
        }),
      }
    }
    const person1 = _getAgesForP('person1')
    const person2 = planParams.people.withPartner
      ? _getAgesForP('person2')
      : null

    return {
      person1,
      person2,
      simulationMonths: block((): ReturnType<typeof _getSimulationMonths> => {
        const withdrawalStartMonthAsMFN = asMFN(withdrawalStartMonth)
        return {
          numMonths,
          lastMonthAsMFN: asMFN(maxMaxAge),
          withdrawalStartMonthAsMFN,
        }
      }),
      validMonthRanges: {
        futureSavingsAsMFN: letIn(
          validMonthRangeAsMFN('future-savings'),
          (orig) => (orig.end < 0 ? null : orig),
        ),
        incomeDuringRetirementAsMFN: letIn(
          validMonthRangeAsMFN('income-during-retirement'),
          (orig) => ({ start: Math.max(orig.start, 0), end: orig.end }),
        ),
        extraSpending: validMonthRangeAsMFN('extra-spending'),
      },
    }
  })
  return {
    v: planParams.v,
    timestamp: planParams.timestamp,
    dialogPositionNominal: planParams.dialogPositionNominal,

    ages,

    wealth: {
      portfolioBalance: planParams.wealth.portfolioBalance,
      futureSavings: normalizeValueForMonthRanges(
        planParams.wealth.futureSavings,
      ),
      incomeDuringRetirement: normalizeValueForMonthRanges(
        planParams.wealth.incomeDuringRetirement,
      ),
    },

    adjustmentsToSpending: {
      extraSpending: {
        essential: normalizeValueForMonthRanges(
          planParams.adjustmentsToSpending.extraSpending.essential,
        ),
        discretionary: normalizeValueForMonthRanges(
          planParams.adjustmentsToSpending.extraSpending.discretionary,
        ),
      },
      tpawAndSPAW: planParams.adjustmentsToSpending.tpawAndSPAW,
    },

    risk: {
      tpaw: planParams.risk.tpaw,
      tpawAndSPAW: planParams.risk.tpawAndSPAW,
      spaw: planParams.risk.spaw,
      spawAndSWR: {
        allocation: normalizeGlidePath(planParams.risk.spawAndSWR.allocation),
      },
      swr: planParams.risk.swr,
    },
    advanced: planParams.advanced,
    results: planParams.results,
  }
}
