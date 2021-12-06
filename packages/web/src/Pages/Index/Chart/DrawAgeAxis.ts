import _ from 'lodash'
import {ChartContext} from './ChartContext'
import {ChartUtils} from './ChartUtils'

export function drawAgeAxis(context: ChartContext) {
  const {ctx, tpawResult, scale} = context
  const tickValues = _.range(
    Math.floor(tpawResult.age.start),
    Math.ceil(tpawResult.age.end) + 1
  )
  ctx.save()
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  ctx.beginPath()

  const [largeTickValues, smallAndMediumTickValues] = _.partition(
    tickValues,
    x => x % 10 === 0
  )
  const [mediumTickValues, smallTickValues] = _.partition(
    smallAndMediumTickValues,
    x => x % 5 === 0
  )

  _drawTickValues(largeTickValues, 'large', context)
  _drawTickValues(mediumTickValues, 'medium', context)
  _drawTickValues(smallTickValues, 'small', context)

  ctx.restore()
}

function _drawTickValues(
  tickValues: number[],
  type: 'small' | 'medium' | 'large',
  {ctx, scale, plotArea}: ChartContext
) {
  const y = scale.curr.y(0)
  const pixelsPerTick = scale.target.x(1) - scale.target.x(0)
  const [tickLength, strokeAndFillStyle, font] =
    type === 'large'
      ? ([15, ChartUtils.color.gray['700'], ChartUtils.getFont(14)] as const)
      : type === 'medium'
      ? ([10, ChartUtils.color.gray['500'], ChartUtils.getFont(11)] as const)
      : ([3, ChartUtils.color.gray['400'], ChartUtils.getFont(8)] as const)
  ctx.font = font
  ctx.fillStyle = strokeAndFillStyle
  tickValues.forEach(age => {
    const x = scale.curr.x(age)

    if (
      Math.abs(x - plotArea.x) > 20 &&
      Math.abs(x - (plotArea.x + plotArea.width)) > 20
    ) {
      ctx.moveTo(x, y)
      ctx.lineTo(x, y + tickLength)
      if (type !== 'small' || pixelsPerTick > 20) {
        ctx.fillText(`${age}`, x, y + tickLength + 2)
      }
    }
  })
  ctx.strokeStyle = strokeAndFillStyle
  ctx.lineWidth = 1
  ctx.stroke()
}
