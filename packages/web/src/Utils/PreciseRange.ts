import _ from 'lodash'

export const preciseRange = (
  start: number,
  end: number,
  stepSize: number,
  precision: number
) => {
  return _.range(_.round((end - start) / stepSize + 1)).map(x =>
    _.round(x * stepSize + start, precision)
  )
}