import { fGet } from '@tpaw/common'
import _ from 'lodash'
import { interpolate } from '../../../../Utils/Interpolate'
import { ChartContext } from '../ChartContext'
import { ChartComponent } from './ChartComponent'

export const chartDrawDataRangeBand = <Data>({
  fillStyle,
  dataFn,
}: {
  fillStyle: string
  dataFn: (data: Data) => {
    min: (x: number) => number
    max: (x: number) => number
  }
}): ChartComponent<Data> => ({
  draw: (context: ChartContext<Data>) => {
    const { canvasContext: ctx, dataTransition, derivedState } = context
    const { scale, plotArea, viewport } = derivedState.curr

    const dataXs = _.range(
      Math.floor(scale.x.inverse(plotArea.x)),
      Math.ceil(scale.x.inverse(plotArea.x + plotArea.width)) + 1,
    )

    ctx.beginPath()
    ctx.rect(
      plotArea.x,
      viewport.y,
      plotArea.width,
      plotArea.y + plotArea.height,
    )
    ctx.clip()

    ctx.fillStyle = fillStyle

    ctx.beginPath()

    const minMaxFromFn = dataFn(dataTransition.from)
    const minMaxTargetFn = dataFn(dataTransition.target)
    const minPixelYs = interpolate({
      from: dataXs.map(minMaxFromFn.min).map(scale.y),
      target: dataXs.map(minMaxTargetFn.min).map(scale.y),
      progress: dataTransition.progress,
    })
    const maxPixelYs = interpolate({
      from: dataXs.map(minMaxFromFn.max).map(scale.y),
      target: dataXs.map(minMaxTargetFn.max).map(scale.y),
      progress: dataTransition.progress,
    })
    const pixelXs = dataXs.map(scale.x)

    ctx.beginPath()
    ctx.moveTo(pixelXs[0], maxPixelYs[0])

    _.zip(pixelXs, maxPixelYs).forEach(([pixelX, maxPixelY]) => {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      ctx.lineTo(pixelX!, maxPixelY!)
    })
    ctx.lineTo(fGet(_.last(pixelXs)), fGet(_.last(minPixelYs)))
    _.zip(pixelXs, minPixelYs)
      .reverse()
      .forEach(([pixelX, maxPixelY]) => {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        ctx.lineTo(pixelX!, maxPixelY!)
      })

    ctx.closePath()
    ctx.fillStyle = fillStyle
    ctx.fill()
  },
})
