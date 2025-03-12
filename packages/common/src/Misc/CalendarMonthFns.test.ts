import { CalendarMonthFns } from './CalendarMonthFns'

describe('CalendarMonthFns', () => {
  test.each([
    [2020, 1, 11, 2020, 12],
    [2020, 1, 12, 2021, 1],
    [2020, 12, 1, 2021, 1],
    [2020, 12, 13, 2022, 1],
    [2020, 2, -1, 2020, 1],
    [2020, 2, -12, 2019, 2],
    [2020, 2, -13, 2019, 1],
    [2020, 2, -14, 2018, 12],
  ])('addMonths', (y, m, dm, ry, rm) => {
    expect(CalendarMonthFns.addMonths({ year: y, month: m }, dm)).toEqual({
      year: ry,
      month: rm,
    })
  })

  test.each([
    [2020, 1, 1, 2020, 2],
    [2020, 1, -1, 2019, 12],
    [2020, 1, -13, 2018, 12],
  ])('getFromMFN', (yn, mn, mfn, yr, mr) => {
    expect(CalendarMonthFns.getFromMFN({ year: yn, month: mn })(mfn)).toEqual({
      year: yr,
      month: mr,
    })
  })

  test.each([
    [2020, 1, -12],
    [2021, 1, 0],
    [2022, 1, 12],
  ])('getToMFN', (year, month, result) => {
    const fn = CalendarMonthFns.getToMFN({ year: 2021, month: 1 })
    expect(fn({ year, month })).toEqual(result)
  })

  test.each([
    [2020, 1, 2021, 12, -23],
    [2020, 1, 2020, 2, -1],
    [2020, 1, 2019, 1, 12],
    [2020, 1, 2018, 1, 24],
  ])('diff', (ya, ma, yb, mb, r) => {
    expect(
      CalendarMonthFns.diff({ year: ya, month: ma }, { year: yb, month: mb }),
    ).toEqual(r)
  })
})
