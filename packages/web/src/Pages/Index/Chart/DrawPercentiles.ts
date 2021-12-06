import _ from 'lodash'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {ChartContext} from './ChartContext'
import {ChartUtils} from './ChartUtils'

export function drawPercentiles(context: ChartContext) {
  const {ctx, tpawResult, percentileInfo, plotArea} = context

  ctx.save()

  ctx.beginPath()
  ChartUtils.roundRect(ctx, plotArea, 25)
  ctx.clip()

  const ageRange = _.range(
    Math.floor(tpawResult.age.start),
    Math.ceil(tpawResult.age.end) + 1
  )

  ctx.lineWidth = 0.5
  ctx.strokeStyle = ChartUtils.color.gray['500']
  _drawLines(context, percentileInfo.backgroundIndexes, ageRange)

  // ctx.lineWidth = 1
  ctx.strokeStyle = ChartUtils.color.gray['900']
  _drawMergingLines(context, percentileInfo.highlightIndexes, ageRange, 1)

  ctx.restore()
}

function _drawLines(
  context: ChartContext,
  indexes: number[],
  ageRange: number[]
) {
  const {ctx, scale} = context

  ctx.beginPath()
  indexes.forEach(percentileIndex => {
    ageRange.forEach((age, i) => {
      const scaledY = _scaledY(context, age, percentileIndex)
      const scaledX = scale.curr.x(age)
      i === 0 ? ctx.moveTo(scaledX, scaledY) : ctx.lineTo(scaledX, scaledY)
    })
  })
  ctx.stroke()
}

function _drawMergingLines(
  context: ChartContext,
  indexes: number[],
  ageRange: number[],
  lineWidth: number
) {
  const {ctx, scale} = context
  ctx.save()
  const ys = ageRange.map((age, i) =>
    indexes.map(i => _scaledY(context, age, i))
  )

  ageRange.reduce((p, c, i) => {
    const y0y1s = indexes.map((_ignore, j) => ({
      y0: ys[i - 1][j],
      y1: ys[i][j],
      extra: 0,
    }))

    indexes.reduceRight((p, c, i) => {
      if (i === 0) return null
      const curr = y0y1s[i]
      const prev = y0y1s[i - 1]
      if (
        Math.round(curr.y0) === Math.round(prev.y0) &&
        Math.round(curr.y1) === Math.round(prev.y1)
      ) {
        prev.extra = curr.extra + 1
        // y0y1s.splice(i, 1)
      }
      return null
    }, null)

    y0y1s.forEach(({y0, y1, extra}) => {
      ctx.lineWidth = lineWidth
      // ctx.lineWidth = lineWidth + extra * lineWidth * 2
      ctx.globalAlpha = linearFnFomPoints(0, 1, 3, 0)(extra)
      ctx.beginPath()
      ctx.moveTo(scale.curr.x(p), y0)
      ctx.lineTo(scale.curr.x(c), y1)
      ctx.stroke()
    })
    return c
  })
  ctx.restore()
}

function _scaledY(
  {tpawResult, scale}: ChartContext,
  age: number,
  percentileIndex: number
) {
  const prevPercentile = tpawResult.prevData[percentileIndex]
  const percentile = tpawResult.data[percentileIndex]
  const interpolatedY = linearFnFomPoints(
    0,
    prevPercentile(age),
    1,
    percentile(age)
  )(tpawResult.dataTransition)
  const scaledY = scale.curr.y(interpolatedY) - 2 // offset by 2 so strokes at bottom are fully visible.
  return scaledY
}
