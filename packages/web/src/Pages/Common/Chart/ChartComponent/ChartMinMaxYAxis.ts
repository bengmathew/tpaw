import _ from 'lodash'
import { ChartContext } from '../ChartContext'
import {
  chartDataTransitionCurrNum,
  chartDataTransitionCurrObj
} from '../ChartUtils/ChartDataTransition'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

const pad = 5
const maxPad = 30
export class ChartMinMaxYAxis<Data> implements ChartComponent<Data> {
  constructor(
    public format: (x: number) => string,
    public fillStyle: string,
    public globalMaxX: (data: Data) => number,
    public minMaxForX: (data: Data, x: number) => {min: number; max: number}
  ) {}

  public draw(context: ChartContext<Data>) {
    const {ctx, stateTransition, currState} = context
    const {scale, plotArea} = currState

    ctx.fillStyle = this.fillStyle
    ctx.font = ChartUtils.getMonoFont(11)

    const drawAtX = (textAlign: 'right' | 'left', graphX: number) => {
      ctx.textAlign = textAlign
      const dataX = scale.x.inverse(graphX)
      const {exactDataY, placementGraphY, exactMeasureHeight} =
        chartDataTransitionCurrObj(stateTransition, ({data, scale}) => {
          const exactDataY = this.minMaxForX(data, dataX)
          const exactMeasure = {
            max: ctx.measureText(this.format(exactDataY.max)),
            min: ctx.measureText(this.format(exactDataY.min)),
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
        max: this.format(exactDataY.max),
        min: this.format(exactDataY.min),
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

    const prevGlobalMaxDataX = this.globalMaxX(stateTransition.prev.data)
    const targetGlobalMaxDataX = this.globalMaxX(stateTransition.target.data)

    const prevGlobalMaxDataY = chartDataTransitionCurrNum(
      stateTransition,
      x => this.minMaxForX(x.data, prevGlobalMaxDataX).max
    )
    const targetGlobalMaxDataY = chartDataTransitionCurrNum(
      stateTransition,
      x => this.minMaxForX(x.data, targetGlobalMaxDataX).max
    )

    const [globalMaxDataY, globalMaxDataX] =
      prevGlobalMaxDataY > targetGlobalMaxDataY
        ? [prevGlobalMaxDataY, prevGlobalMaxDataX]
        : [targetGlobalMaxDataY, targetGlobalMaxDataX]
    const globalMaxDataYText = this.format(globalMaxDataY)
    const globalMaxGraphX = scale.x(globalMaxDataX)
    const globalMaxGraphY = scale.y(globalMaxDataY)
    if (
      globalMaxGraphY <
      Math.min(leftInfo.maxGraphY, rightInfo.maxGraphY) - 50
    ) {
      const globalMaxYTextMeasure = ctx.measureText(this.format(1))
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
