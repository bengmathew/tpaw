import { linearFnFomPoints } from '../../../../Utils/LinearFn'

// TODO: Make private to ChartDataTransition?
export const zeroOneInterpolate = (
  atZero: number,
  atOne: number,
  {transition}:{transition: number}
) => linearFnFomPoints(0, atZero, 1, atOne)(transition)

