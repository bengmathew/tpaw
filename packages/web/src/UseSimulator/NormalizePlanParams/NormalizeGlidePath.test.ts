import {
  CalendarMonthFns,
  GlidePath,
  Month,
  assert,
  generateSmallId,
} from '@tpaw/common'
import _ from 'lodash'
import { PickType } from '../../Utils/UtilityTypes'
import {
  NormalizedGlidePath,
  NormalizedGlidePathEntry,
  normalizeGlidePath,
  normalizedGlidePathErrorMsg,
} from './NormalizeGlidePath'
import { normalizedMonthErrorMsg } from './NormalizeLabeledAmountTimedList/NormalizedMonth'
import {
  getFromMFNToNumericAge,
  getToMFN,
  normalizePlanParamsAges,
} from './NormalizePlanParamsAges'

describe('NormalizeGlidePath', () => {
  const nowAs = { calendarMonth: { year: 2024, month: 3 } }
  const mfnToCalendarMonth = CalendarMonthFns.getFromMFN(nowAs.calendarMonth)
  const ages = normalizePlanParamsAges(
    {
      withPartner: true,
      person1: {
        ages: {
          type: 'retiredWithNoRetirementDateSpecified' as const,
          monthOfBirth: mfnToCalendarMonth(-10),
          maxAge: { inMonths: 20 },
        },
      },
      person2: {
        ages: {
          type: 'retiredWithNoRetirementDateSpecified' as const,
          monthOfBirth: mfnToCalendarMonth(-10),
          maxAge: { inMonths: 30 },
        },
      },
      withdrawalStart: 'person2',
    },
    nowAs.calendarMonth,
  )
  const toMFN = getToMFN({ nowAs, ages })

  const m = (mfn: number): Month => ({
    type: 'numericAge',
    person: 'person1',
    age: { inMonths: getFromMFNToNumericAge({ ages }).person1(mfn) },
  })
  const i = (
    month: Month,
    stocks: number,
    issue: 'duplicate' | 'atOrPastEnd' | 'none',
    error: 'atOrPastEnd' | 'duplicate' | 'pastMaxAge' | 'none',
  ) => ({
    id: generateSmallId(),
    month,
    stocks,
    issue,
    error,
  })

  test.each([
    [0, 0.5, 0.5, [i(m(1), 0.5, 'none', 'none')], 0.5],
    [
      0,
      0.5,
      0.5,
      [i(m(1), 0.5, 'none', 'none'), i(m(1), 0.6, 'duplicate', 'duplicate')],
      0.5,
    ],
    [
      0,
      0.5,
      0.5,
      [
        i(m(15), 0.5, 'none', 'pastMaxAge'),
        i(m(15), 0.6, 'duplicate', 'duplicate'),
      ],
      0.5,
    ],
    [0, 0.5, 0.5, [], 0.5],
    [0, 0.5, 0.5, [i(m(200), 0.9, 'atOrPastEnd', 'atOrPastEnd')], 0.5],
    [0, 0.5, 0.5, [i(m(15), 0.9, 'none', 'pastMaxAge')], 0.5],
    [
      -2,
      0.5,
      0.5,
      [i(m(-1), 0.6, 'none', 'none'), i(m(1), 0.8, 'none', 'none')],
      0.7,
    ],
  ])(
    '',
    (startMFN, startStocks, endStocks, intermediateIn, resultNowStocks) => {
      const intermediate = intermediateIn.map((x, i) => ({
        ...x,
        indexToSortByAdded: i,
      }))
      const normEntry = (
        x: (typeof intermediate)[0],
      ): NormalizedGlidePathEntry => {
        const asMFN = toMFN.forMonth.pastElided(x.month)
        assert(asMFN !== 'inThePast')
        const personMaxAgeAsMFN = (personType: 'person1' | 'person2') =>
          toMFN.forMonth.fNotInPast({
            type: 'namedAge',
            person: personType,
            age: 'max',
          })
        return {
          id: x.id,
          ignore: x.issue !== 'none',
          month: {
            type: 'normalizedMonth',
            isInThePast: false,
            asMFN,
            baseValue: x.month,
            validRangeAsMFN: {
              includingLocalConstraints: {
                start: 1,
                end: personMaxAgeAsMFN('person1'),
              },
              excludingLocalConstraints: {
                start: 1,
                end: personMaxAgeAsMFN('person2') - 1,
              },
            },
            errorMsg:
              x.error === 'none'
                ? null
                : x.error === 'pastMaxAge'
                  ? normalizedMonthErrorMsg.pastMaxAge.person1
                  : normalizedGlidePathErrorMsg[x.error],
          },
          stocks: x.stocks,
          indexToSortByAdded: x.indexToSortByAdded,
        }
      }
      expect(
        normalizeGlidePath(
          {
            start: { month: mfnToCalendarMonth(startMFN), stocks: startStocks },
            intermediate: _.fromPairs(
              intermediate
                .map((x, i): GlidePath['intermediate']['string'] => ({
                  id: x.id,
                  indexToSortByAdded: x.indexToSortByAdded,
                  month: x.month,
                  stocks: x.stocks,
                }))
                .map((x) => [x.id, x]),
            ),
            end: { stocks: endStocks },
          },
          toMFN,
          ages,
        ),
      ).toEqual({
        nowAsCalendarMonth: nowAs.calendarMonth,
        now: { stocks: resultNowStocks },
        intermediate: intermediate
          .filter(
            (x) =>
              x.issue !== 'atOrPastEnd' &&
              toMFN.forMonth.pastElided(x.month) !== 'inThePast',
          )
          .map(normEntry),
        end: { stocks: endStocks },
        atOrPastEnd: intermediate
          .filter((x) => x.issue === 'atOrPastEnd')
          .map(normEntry),
      } as NormalizedGlidePath)
    },
  )

})