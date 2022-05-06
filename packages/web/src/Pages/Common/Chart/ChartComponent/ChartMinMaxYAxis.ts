import _ from 'lodash'
import {ChartContext} from '../ChartContext'
import {
  chartDataTransitionCurrNum,
  chartDataTransitionCurrObj,
} from '../ChartUtils/ChartDataTransition'
import {ChartUtils} from '../ChartUtils/ChartUtils'
import {ChartComponent} from './ChartComponent'

const pad = 5
const maxPad = 30
export class ChartMinMaxYAxis<Data> implements ChartComponent<Data> {
  constructor(
    public format: (data: Data, x: number) => string,
    public fillStyle: string,
    public globalMaxX: (data: Data) => number,
    public minMaxForX: (data: Data, x: number) => {min: number; max: number}
  ) {}

  public draw(context: ChartContext<Data>) {
    const {canvasContext: ctx, dataTransition, derivedState} = context
    const {scale, plotArea} = derivedState.curr

    ctx.fillStyle = this.fillStyle
    ctx.font = ChartUtils.getMonoFont(11)

    const format = (x: number) => this.format(dataTransition.target, x)

    const drawAtX = (textAlign: 'right' | 'left', graphX: number) => {
      ctx.textAlign = textAlign
      const dataX = scale.x.inverse(graphX)
      const {exactDataY, placementGraphY, exactMeasureHeight} =
        chartDataTransitionCurrObj(dataTransition, data => {
          const exactDataY = this.minMaxForX(data, dataX)
          const exactMeasure = {
            max: ctx.measureText(format(exactDataY.max)),
            min: ctx.measureText(format(exactDataY.min)),
          }
          const testPoints = (labelWidth: number) => {
            const sign = textAlign === 'left' ? 1 : -1
            return _.times(11, i => graphX + (i * labelWidth * sign) / 10).map(
              x => scale.x.inverse(x)
            )
          }
          const placementDataY = {
            max: Math.max(
              ...testPoints(exactMeasure.max.width).map(
                dataX => this.minMaxForX(data, dataX).max
              )
            ),
            min: Math.min(
              ...testPoints(exactMeasure.min.width).map(
                dataX => this.minMaxForX(data, dataX).min
              )
            ),
          }

          const placementGraphY = {
            max: scale.y(placementDataY.max),
            min: scale.y(placementDataY.min),
          }
          return {
            exactDataY,
            placementGraphY,
            exactMeasureHeight: {
              min: exactMeasure.min.actualBoundingBoxAscent,
              max: exactMeasure.max.actualBoundingBoxAscent,
            },
          }
        })
      const exactGraphY = {
        max: scale.y(exactDataY.max),
        min: scale.y(exactDataY.min),
      }
      const exactDataYStr = {
        max: format(exactDataY.max),
        min: format(exactDataY.min),
      }

      const idealMinLabelGraphY = exactGraphY.min + exactMeasureHeight.min + pad
      const actualMinLabelGraphY = Math.min(
        plotArea.y + plotArea.height - 2,
        exactGraphY.min + exactMeasureHeight.min + maxPad,
        placementGraphY.min + exactMeasureHeight.min + pad
      )

      const idealMaxLabelGraphY = exactGraphY.max - pad
      const actualMaxLabelGraphY = Math.max(
        exactGraphY.max - maxPad,
        placementGraphY.max - pad
      )

      let adjActualMaxLabelGraphY: null | number = null

      if (exactDataYStr.max !== exactDataYStr.min) {
        adjActualMaxLabelGraphY = Math.min(
          actualMaxLabelGraphY,
          actualMinLabelGraphY - exactMeasureHeight.min - 2
        )
        ctx.fillText(exactDataYStr.min, graphX, actualMinLabelGraphY)
        ctx.fillText(exactDataYStr.max, graphX, adjActualMaxLabelGraphY)
      } else {
        const maxIsBetter =
          Math.abs(actualMaxLabelGraphY - idealMaxLabelGraphY) <=
          Math.abs(actualMinLabelGraphY - idealMinLabelGraphY)
        const minHitBottom = actualMinLabelGraphY < idealMinLabelGraphY
        adjActualMaxLabelGraphY =
          maxIsBetter || minHitBottom
            ? actualMaxLabelGraphY
            : actualMinLabelGraphY
        ctx.fillText(exactDataYStr.max, graphX, adjActualMaxLabelGraphY)
      }
      return {graphX, maxGraphY: adjActualMaxLabelGraphY}
    }
    const leftInfo = drawAtX('left', plotArea.x)
    const rightInfo = drawAtX('right', plotArea.x + plotArea.width)
    ctx.globalAlpha = 1

    // The global max.
    ctx.textAlign = 'center'

    const prevGlobalMaxDataX = this.globalMaxX(dataTransition.prev)
    const targetGlobalMaxDataX = this.globalMaxX(dataTransition.target)

    const prevGlobalMaxDataY = chartDataTransitionCurrNum(
      dataTransition,
      x => this.minMaxForX(x, prevGlobalMaxDataX).max
    )
    const targetGlobalMaxDataY = chartDataTransitionCurrNum(
      dataTransition,
      x => this.minMaxForX(x, targetGlobalMaxDataX).max
    )

    const [globalMaxDataY, globalMaxDataX] =
      prevGlobalMaxDataY > targetGlobalMaxDataY
        ? [prevGlobalMaxDataY, prevGlobalMaxDataX]
        : [targetGlobalMaxDataY, targetGlobalMaxDataX]
    const globalMaxDataYText = format(globalMaxDataY)
    const globalMaxGraphX = scale.x(globalMaxDataX)
    const globalMaxGraphY = scale.y(globalMaxDataY)
    if (
      globalMaxGraphY <
      Math.min(leftInfo.maxGraphY, rightInfo.maxGraphY) - 50
    ) {
      const globalMaxYTextMeasure = ctx.measureText(format(1))
      const globalMaxGraphXAdj = _.clamp(
        globalMaxGraphX,
        plotArea.x,
        plotArea.x + plotArea.width - globalMaxYTextMeasure.width
      )
      ctx.fillText(
        globalMaxDataYText,
        globalMaxGraphXAdj,
        globalMaxGraphY - pad
      )
    }
  }
}
