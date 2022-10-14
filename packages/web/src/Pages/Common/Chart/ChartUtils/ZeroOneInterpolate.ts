export const zeroOneInterpolate = (
  atZero: number,
  atOne: number,
  {transition}: {transition: number}
) => atZero + (atOne - atZero) * transition
