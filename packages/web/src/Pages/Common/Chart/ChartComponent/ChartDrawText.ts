import {ChartSizing} from '../Chart'
import {ChartComponent} from './ChartComponent'

export const chartDrawText = <Data>(
  paramsFn: (
    data: Data,
    sizing: ChartSizing
  ) => {
    font: string
    fillStyle: string
    text: string
    textAlign: CanvasTextAlign
    position: {graphX: number; graphY: number}
  }
): ChartComponent<Data> => ({
  draw: ctx => {
    const {canvasContext, dataTransition, sizing} = ctx
    const {text, position, textAlign, font, fillStyle} = paramsFn(
      dataTransition.target,
      sizing
    )
    canvasContext.fillStyle = fillStyle
    canvasContext.font = font
    canvasContext.textAlign = textAlign
    canvasContext.fillText(text, position.graphX, position.graphY)
  },
})
