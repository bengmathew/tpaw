export const joinWithCommaAnd = (x: string[]) =>
  x.length === 0
    ? ''
    : x.length === 1
    ? x[0]
    : x.length === 2
    ? x.join(' and ')
    : `${x.slice(0, x.length - 1).join(', ')}, and ${x[x.length - 1]}`
