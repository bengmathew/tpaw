import {
  normalizedMonthErrorMsg,
  normalizedMonthRangeCheckAndSquishRangeForAge,
} from './NormalizedMonth'

describe('NormalizeMonth', () => {
  //TODO:
  // describe('normalizedMonthRangeCheckAndSquishRangeForAge', () => {
  //   describe('no squish', () => {
  //     test('no error', () => {
  //       const data = {
  //         asMFN: 2,
  //         baseValue: {
  //           type: 'calendarMonth' as const,
  //           calendarMonth: { year: 0, month: 0 },
  //         },
  //         validRangeAsMFN: {
  //           includingLocalConstraints: { start: 0, end: 2 },
  //           excludingLocalConstraints: { start: 0, end: 2 },
  //         },
  //       }
  //       expect(
  //         normalizedMonthRangeCheckAndSquishRangeForAge(data, {
  //           person1: { maxAge: { asMFN: 1 } },
  //           person2: null,
  //         }),
  //       ).toEqual({
  //         type: 'normalizedMonth',
  //         isInThePast: false,
  //         ...data,
  //         errorMsg: null,
  //       })
  //     })

  //     test('out of range', () => {
  //       const data = {
  //         asMFN: 2,
  //         baseValue: {
  //           type: 'calendarMonth' as const,
  //           calendarMonth: { year: 0, month: 0 },
  //         },
  //         validRangeAsMFN: {
  //           includingLocalConstraints: { start: 0, end: 1 },
  //           excludingLocalConstraints: { start: 0, end: 2 },
  //         },
  //       }
  //       expect(
  //         normalizedMonthRangeCheckAndSquishRangeForAge(data, {
  //           person1: { maxAge: { asMFN: 1 } },
  //           person2: null,
  //         }),
  //       ).toEqual({
  //         type: 'normalizedMonth',
  //         isInThePast: false,
  //         ...data,
  //         errorMsg: normalizedMonthErrorMsg.outOfRange,
  //       })
  //     })
  //   })

  //   describe('squish', () => {
  //     test('no error', () => {
  //       const data = {
  //         asMFN: 2,
  //         baseValue: {
  //           type: 'numericAge' as const,
  //           person: 'person1' as const,
  //           age: { inMonths: 3 },
  //         },
  //         validRangeAsMFN: {
  //           includingLocalConstraints: { start: 0, end: 5 },
  //           excludingLocalConstraints: { start: 0, end: 6 },
  //         },
  //       }
  //       expect(
  //         normalizedMonthRangeCheckAndSquishRangeForAge(data, {
  //           person1: { maxAge: { asMFN: 4 } },
  //           person2: null,
  //         }),
  //       ).toEqual({
  //         type: 'normalizedMonth',
  //         isInThePast: false,
  //         ...data,
  //         validRangeAsMFN: {
  //           ...data.validRangeAsMFN,
  //           includingLocalConstraints: { start: 0, end: 4 },
  //         },
  //         errorMsg: null,
  //       })
  //     })

  //     test('past max age', () => {
  //       const data = {
  //         asMFN: 4,
  //         baseValue: {
  //           type: 'numericAge' as const,
  //           person: 'person1' as const,
  //           age: { inMonths: 4 },
  //         },
  //         validRangeAsMFN: {
  //           includingLocalConstraints: { start: 0, end: 5 },
  //           excludingLocalConstraints: { start: 0, end: 10 },
  //         },
  //       }
  //       expect(
  //         normalizedMonthRangeCheckAndSquishRangeForAge(data, {
  //           person1: { maxAge: { asMFN: 3 } },
  //           person2: null,
  //         }),
  //       ).toEqual({
  //         type: 'normalizedMonth',
  //         isInThePast: false,
  //         ...data,
  //         validRangeAsMFN: {
  //           ...data.validRangeAsMFN,
  //           includingLocalConstraints: { start: 0, end: 3 },
  //         },
  //         errorMsg: normalizedMonthErrorMsg.pastMaxAge.person1,
  //       })
  //     })

  //     test('out of range', () => {
  //       const data = {
  //         asMFN: 6,
  //         baseValue: {
  //           type: 'numericAge' as const,
  //           person: 'person1' as const,
  //           age: { inMonths: 4 },
  //         },
  //         validRangeAsMFN: {
  //           includingLocalConstraints: { start: 0, end: 5 },
  //           excludingLocalConstraints: { start: 0, end: 10 },
  //         },
  //       }
  //       expect(
  //         normalizedMonthRangeCheckAndSquishRangeForAge(data, {
  //           person1: { maxAge: { asMFN: 3 } },
  //           person2: null,
  //         }),
  //       ).toEqual({
  //         type: 'normalizedMonth',
  //         isInThePast: false,
  //         ...data,
  //         validRangeAsMFN: {
  //           ...data.validRangeAsMFN,
  //           includingLocalConstraints: { start: 0, end: 3 },
  //         },
  //         errorMsg: normalizedMonthErrorMsg.outOfRange,
  //       })
  //     })
  //   })
  // })
})
