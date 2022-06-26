import _ from 'lodash'

export function stats(data:Float64Array)  {
  let n = data.length
  let mean = _.mean(data) 
  let variance = _.mean(data
      .map(value=> {
          let diff = mean - value;
          return Math.pow(diff, 2)
      }))
      
  let standardDeviation = Math.sqrt(variance);
  return  {mean, variance, standardDeviation, n}
}