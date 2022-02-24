/* eslint-disable @typescript-eslint/no-unsafe-argument */
import _ from 'lodash'
import {zeroOneInterpolate} from './ZeroOneInterpolate'

export type ChartDataTransition<Data> = {
  prev: Data
  target: Data
  transition: number
}

export const chartDataTransitionCurrObj = <
  Obj extends Record<string, number | Record<string, number>>,
  Data
>(
  dataTransition: ChartDataTransition<Data>,
  dataFn: (state: Data) => Obj
): {[P in keyof Obj]: Obj[P]} => {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.prev)
  return _.mapValues(target, (value, key) =>
    typeof value === 'number'
      ? zeroOneInterpolate(
          prev[key] as number,
          target[key] as number,
          dataTransition
        )
      : chartDataTransitionCurrObj(
          {
            prev: prev[key] as any,
            target: target[key] as any,
            transition: dataTransition.transition,
          },
          (x: any) => x
        )
  ) as any
}

export function chartDataTransitionCurrNum<Data>(
  dataTransition: ChartDataTransition<Data>,
  dataFn: (state: Data) => number
): number {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.prev)
  return zeroOneInterpolate(prev, target, dataTransition)
}

export function chartDataTransitionCurrNumArr<Data>(
  dataTransition: ChartDataTransition<Data>,
  dataFn: (state: Data) => number[]
): number[] {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.prev)
  return target.map((t, i) => zeroOneInterpolate(prev[i], t, dataTransition))
}

// TODO: Remove
export const chartDataTransitionNew = (
  target: number,
  curr: ChartDataTransition<number>
): ChartDataTransition<number> => ({
  prev: chartDataTransitionCurrNum(curr, x => x),
  target,
  transition: 0,
})
