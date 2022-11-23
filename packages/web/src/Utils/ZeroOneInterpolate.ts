export const zeroOneInterpolate = (
  atZero: number,
  atOne: number,
  progress: number,
) => atZero + (atOne - atZero) * progress
