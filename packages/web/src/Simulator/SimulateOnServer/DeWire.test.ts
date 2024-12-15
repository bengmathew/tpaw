import { deWire } from './DeWire'

describe('deWire', () => {
  test('general', () => {
    expect(
      deWire({
        a: {
          b: 50,
          cX100: 50,
          dX1000: [50],
          u: {
            $case: 'ok',
            timestampMs: 123,
            startTimestampMs: 100,
            oOpt: undefined,
          },
        },
      }),
    ).toEqual({
      a: {
        b: 50,
        c: 50 * (1 / 100),
        d: [50 * (1 / 1000)],
        u: {
          type: 'ok',
          timestamp: 123,
          startTimestamp: 100,
          o: null,
        },
      },
    })
    expect(deWire({ a: [{ bX100: 50 }] })).toEqual({
      a: [{ b: 50 * (1 / 100) }],
    })
    expect(() =>
      deWire({ a: { b: undefined } }, { scale: 2, isOpt: false }),
    ).toThrow()
  })
})

const fn = (x: number) => x
