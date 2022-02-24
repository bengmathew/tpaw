
export const noNegZero = (x: number) => (Object.is(x, -0) ? 0 : x)