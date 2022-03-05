import _ from 'lodash'
import {assert} from '../Utils/Utils'

export namespace StatsTools {
  export function pivot<T>(rows: T[][]) {
    if (rows.length === 0) return rows
    const numColsArr = _.uniq(rows.map(x => x.length))
    assert(numColsArr.length === 1)
    const numCols = numColsArr[0]
    return _.range(numCols).map(colIndex => rows.map(row => row[colIndex]))
  }

  export function pickPercentilesFromSorted(
    sortedData: Float64Array[],
    percentiles: number[]
  ) {
    if (sortedData.length === 0) return [] as number[][]
    const indices = percentiles.map(percentile =>
      Math.round((percentile * sortedData[0].length) / 100)
    )
    return sortedData.map(row => indices.map(i => row[i]))
  }
}
