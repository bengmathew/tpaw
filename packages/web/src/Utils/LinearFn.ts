export const linearFnFomPoints = (
  x0: number,
  y0: number,
  x1: number,
  y1: number
) => {
  const slope = (y1 - y0) / (x1 - x0)
  return linearFnFromPointAndSlope(x0, y0, slope)
}

export const linearFnFromPointAndSlope = (
  x: number,
  y: number,
  slope: number
) => {
  const intercept = y - slope * x
  return linearFnFromSlopeAndIntercept(slope, intercept)
}

export type LinearFn = ReturnType<typeof linearFnFromSlopeAndIntercept>
export const linearFnFromSlopeAndIntercept = (
  slope: number,
  intercept: number
) => {
  const result = (x: number) => slope * x + intercept
  result.inverse = (y: number) => (y - intercept) / slope
  return result
}
