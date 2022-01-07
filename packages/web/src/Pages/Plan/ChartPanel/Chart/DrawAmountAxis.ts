import _ from 'lodash'
import { fGet } from '../../../../Utils/Utils'
import { ChartContext } from './ChartContext'
import { ChartUtils } from './ChartUtils'

export function drawAmountAxis(chart: ChartContext) {
  const {maxY} = chart
  const tickValues = _axisTicksValues(maxY.target, maxY.curr)

  _impl(chart, 'small', tickValues.small)
  _impl(chart, 'large', tickValues.large)
}

function _impl(
  {ctx, scale, plotArea}: ChartContext,
  type: 'large' | 'small',
  tickValues: number[]
) {
  if(tickValues.length === 0) return
  const fontSize = type === 'large' ? 14 : 8
  ctx.font = ChartUtils.getFont(fontSize)
  if (type === 'large') {
    ctx.fillStyle = ChartUtils.color.gray[700]
    ctx.strokeStyle = ChartUtils.color.gray[700]
  } else {
    ctx.fillStyle = ChartUtils.color.gray[500]
    ctx.strokeStyle = ChartUtils.color.gray[500]
  }
  ctx.textAlign = 'end'
  ctx.textBaseline = 'middle'
  ctx.beginPath()
  const tickLength = type === 'large' ? 15 : 6

  const pixelsPerTick =
    (scale.target.y(fGet(_.last(tickValues))) - scale.target.y(tickValues[0])) /
    tickValues.length

  const showTickLabel = tickValues.length === 1 || Math.abs(pixelsPerTick) > 19

  tickValues.forEach((tickValue, i) => {
    const y = scale.curr.y(tickValue)
    if (
      Math.abs(y - plotArea.y) > 20 &&
      Math.abs(y - (plotArea.y + plotArea.height)) > 20
    ) {
      ctx.moveTo(plotArea.x - tickLength, y)
      ctx.lineTo(plotArea.x, y)
      if (showTickLabel) {
        const tickLabel = `${Math.round(tickValue / 1000)}${
          type === 'large' ? 'K' : ''
        }`
        ctx.fillText(tickLabel, plotArea.x - tickLength - 3, y)
      }
    }
  })
  ctx.lineWidth = 1
  ctx.stroke()
}

function _axisTicksValues(targetMax: number, max: number) {
  const byType = (type: 'large' | 'small') => {
    const size = _axisTicksSize(targetMax, type)
    const numTicks = Math.floor(max / size)
    return _.range(numTicks).map(x => Math.round((x + 1) * size))
  }
  const small = byType('small')
  const large = byType('large')
  return {large, small: _.difference(small, large)}
}

function _axisTicksSize(max: number, type: 'large' | 'small') {
  const normalizedOne = Math.pow(10, _numDigits(max) - 1)
  const normaliized = _axisTicksSizeNormalized(max / normalizedOne, type)
  return normaliized * normalizedOne
}

function _axisTicksSizeNormalized(
  normalizedMax: number,
  type: 'large' | 'small'
) {
  if (normalizedMax < 6.5) return type === 'large' ? 1.0 : 0.1
  return type === 'large' ? 1 : 0.2
}

function _numDigits(x: number) {
  return (Math.log10((x ^ (x >> 31)) - (x >> 31)) | 0) + 1
}
