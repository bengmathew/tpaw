import _ from 'lodash'

export const getNetPresentValue = (
  rate: number,
  amounts: number[],
  {
    amountsStartAtCurrentYear = false,
  }: {amountsStartAtCurrentYear?: boolean} = {}
) =>
  // Significantly faster than Math.pow(), _.sum, and _.map.
  amounts.reduce(
    (p, amount) => {
      const newRate = p.rate * (1 + rate)
      return {rate: newRate, sum: p.sum + amount / newRate}
    },
    {rate: 1, sum: 0}
  ).sum * (amountsStartAtCurrentYear ? 1 + rate : 1)

export const getNetPresentValueArr = (r: number, amounts: number[]) => {
  const result = _.times(amounts.length)
  amounts.reduceRight((p, c, i) => {
    const pv = c + p / (1 + r)
    result[i] = pv
    return pv
  }, 0)
  return result
}

