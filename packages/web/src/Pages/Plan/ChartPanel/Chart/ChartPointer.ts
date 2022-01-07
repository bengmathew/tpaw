import {gsap} from 'gsap'
import _ from 'lodash'
import {formatCurrency} from '../../../../Utils/FormatCurrency'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {fGet} from '../../../../Utils/Utils'
import {ChartContext} from './ChartContext'
import {ChartUtils} from './ChartUtils'

export class ChartPointer {
  private _position
  private _onDraw
  private _ageInfo: {target: number; curr: number} | null = null
  private _boxYsInfo: {target: number; curr: number}[] | null = null
  private _labelSideInfo = {target: 1, curr: 1}
  private _transition = {curr: 0}
  // private _pointAnimation: ReturnType<typeof gsap['to']> | null = null
  private _labelSideAnimation: ReturnType<typeof gsap['to']> | null = null
  private _boxYsAnimation: ReturnType<typeof gsap['timeline']> | null = null
  private _ageAnimation: ReturnType<typeof gsap['to']> | null = null
  private _transitionAnimation: ReturnType<typeof gsap['to']> | null = null

  constructor(onDraw: () => void) {
    this._position = {x: 0, y: 0}
    this._onDraw = onDraw
  }

  destroy() {
    this._ageAnimation?.kill()
    this._labelSideAnimation?.kill()
    this._boxYsAnimation?.kill()
    this._transitionAnimation?.kill()
  }
  show() {
    this._transitionAnimation = gsap.to(this._transition, {
      curr: 1,
      duration: 1,
      onUpdate: this._onDraw,
      ease: 'power4',
    })
  }
  hide() {
    this._transitionAnimation = gsap.to(this._transition, {
      curr: 0,
      duration: 1,
      onUpdate: this._onDraw,
      ease: 'power4',
    })
  }

  setPosition(position: {x: number; y: number}) {
    this._position = {...position}
  }

  draw(chartContext: ChartContext) {
    const {ctx, scale, tpawResult, percentileInfo, plotArea, maxY} =
      chartContext
    const {highlightIndexes} = percentileInfo
    ctx.save()
    ctx.globalAlpha = this._transition.curr
    const targetAge = _.clamp(
      Math.round(scale.curr.x.inverse(this._position.x)),
      Math.ceil(tpawResult.age.start),
      Math.floor(tpawResult.age.end)
    )
    if (!this._ageInfo) this._ageInfo = {target: targetAge, curr: targetAge}
    if (this._ageInfo.target !== targetAge) {
      if (this._ageAnimation) this._ageAnimation.kill()
      this._ageInfo.target = targetAge
      this._ageAnimation = gsap.to(this._ageInfo, {
        curr: targetAge,
        duration: 1,
        onUpdate: this._onDraw,
        ease: 'power4',
      })
    }

    const age = this._ageInfo.curr
    const ageX = scale.curr.x(age)
    const targetAgeX = scale.curr.x(targetAge)

    ctx.font = ChartUtils.getMonoFont(11)
    ctx.textAlign = 'left'
    const texts = highlightIndexes.map(i => ({
      left: `${tpawResult.src.args.percentiles[i]}th`,
      middle: '-',
      right: `${formatCurrency(tpawResult.data[i](targetAge))}`,
    }))
    const maxWidthMeasureLeft = fGet(
      _.maxBy(
        texts.map(text => ctx.measureText(text.left)),
        x => x.width
      )
    )
    const maxWidthMeasureMiddle = fGet(
      _.maxBy(
        texts.map(text => ctx.measureText(text.middle)),
        x => x.width
      )
    )
    const maxWidthMeasureRight = fGet(
      _.maxBy(
        texts.map(text => ctx.measureText(text.right)),
        x => x.width
      )
    )

    const offsetX = 30
    const padding = {y: 7, x: 15}
    const gapY = 0
    const textGapX = 4

    const boxWithoutY = {
      x: ageX + offsetX,
      width:
        maxWidthMeasureLeft.width +
        textGapX +
        maxWidthMeasureMiddle.width +
        textGapX +
        maxWidthMeasureRight.width +
        padding.x,
      height:
        maxWidthMeasureLeft.actualBoundingBoxAscent +
        maxWidthMeasureLeft.actualBoundingBoxDescent +
        padding.y,
    }

    const pointYs = highlightIndexes.map((highlightI, i) => {
      const interpolatedY = linearFnFomPoints(
        0,
        tpawResult.prevData[highlightI](age),
        1,
        tpawResult.data[highlightI](age)
      )(tpawResult.dataTransition)
      // -2 because percentiles are offset.
      return scale.curr.y(interpolatedY) - 2
    })

    // const targetBoxYs = (() => {
    //   let boxYs = highlightIndexes.map((highlightI, i) =>
    //     Math.max(
    //       scale.target.y(maxY.target),
    //       scale.target.y(tpawResult.data[highlightI](targetAge)) -
    //         20 -
    //         (boxWithoutY.height - padding.y / 2)
    //     )
    //   )

    //   boxYs.reduceRight((a, y, i) => {
    //     if (i === boxYs.length - 1) return null
    //     boxYs[i] = Math.max(boxYs[i + 1] + boxWithoutY.height + gapY, y)
    //     return null
    //   }, null)

    //   boxYs.forEach((y, i) => {
    //     const prevTop = i === 0 ? plotArea.y + plotArea.height : boxYs[i - 1]
    //     boxYs[i] = Math.min(prevTop - gapY - boxWithoutY.height, y)
    //   })
    //   return boxYs
    // })()
    const targetBoxYs = (() => {
      let pointRange = {
        end: scale.target.y(tpawResult.data[highlightIndexes[0]](targetAge)),
        start: scale.target.y(
          tpawResult.data[fGet(_.last(highlightIndexes))](targetAge)
        ),
      }
      const height = boxWithoutY.height * highlightIndexes.length
      const top = Math.max(
        plotArea.y,
        pointRange.start + (pointRange.end - height - pointRange.start) * 0.8
      )

      return highlightIndexes.map(
        (h, i) => top + height - (i + 1) * boxWithoutY.height
      )
    })()
    if (!this._boxYsInfo)
      this._boxYsInfo = targetBoxYs.map(y => ({target: y, curr: y}))
    if (
      !_.isEqual(
        this._boxYsInfo.map(x => x.target),
        targetBoxYs
      )
    ) {
      if (this._boxYsAnimation) this._boxYsAnimation.kill()
      targetBoxYs.forEach((y, i) => (fGet(this._boxYsInfo)[i].target = y))
      this._boxYsAnimation = gsap.timeline({onUpdate: this._onDraw})
      targetBoxYs.forEach((y, i) => {
        fGet(this._boxYsAnimation).to(
          fGet(this._boxYsInfo)[i],
          {curr: y, duration: 1, ease: 'power4'},
          0
        )
      })
    }

    const targetLabelSide =
      boxWithoutY.x + boxWithoutY.width + 20 > plotArea.x + plotArea.width
        ? -1
        : 1
    if (this._labelSideInfo.target !== targetLabelSide) {
      if (this._labelSideAnimation) this._labelSideAnimation.kill()
      this._labelSideInfo.target = targetLabelSide
      this._labelSideAnimation = gsap.to(this._labelSideInfo, {
        curr: targetLabelSide,
        duration: 1,
        onUpdate: this._onDraw,
        ease: 'power4',
      })
    }
    boxWithoutY.x = linearFnFomPoints(
      1,
      boxWithoutY.x,
      -1,
      ageX - offsetX - boxWithoutY.width
    )(this._labelSideInfo.curr)

    const boxHandleX = linearFnFomPoints(
      1,
      boxWithoutY.x,
      -1,
      boxWithoutY.x + boxWithoutY.width
    )(this._labelSideInfo.curr)

    const bubbleBox = {
      x: boxWithoutY.x,
      y: fGet(_.last(this._boxYsInfo)).curr - padding.y / 2,
      width: boxWithoutY.width,
      height: boxWithoutY.height * highlightIndexes.length + padding.y,
    }

    highlightIndexes.forEach((highlightI, i) => {
      const box = {...boxWithoutY, y: fGet(this._boxYsInfo)[i].curr}

      ctx.save()
      ctx.globalAlpha = 0.6

      // Draw the target.
      ctx.beginPath()
      ctx.ellipse(ageX, pointYs[i], 4, 4, 0, 0, Math.PI * 4)
      ctx.lineWidth = 2
      ctx.fillStyle = ChartUtils.color.gray[1000]
      ctx.fill()

      // Draw the line from target to bubble.
      ctx.beginPath()
      ctx.moveTo(ageX, pointYs[i])
      // ctx.lineTo(boxHandleX- this._labelSideInfo.curr *7, box.y + box.height * 0.5)
      // ctx.lineTo(boxHandleX, box.y + box.height * 0.5)
      ctx.bezierCurveTo(
        ageX + offsetX * 0.35 * this._labelSideInfo.curr,
        pointYs[i],
        boxHandleX - offsetX * .5 * this._labelSideInfo.curr,
        box.y + box.height * 0.5,
        boxHandleX,
        box.y + box.height * 0.5
      )

      ctx.lineWidth = 1
      ctx.strokeStyle = ChartUtils.color.gray[1000]
      ctx.stroke()

      // // Draw the target.
      // ctx.beginPath()
      // ctx.ellipse(boxHandleX, box.y + box.height * 0.5, 2, 2, 0, 0, Math.PI * 4)
      // ctx.lineWidth = 2
      // ctx.fillStyle = ChartUtils.color.gray[1000]
      // ctx.fill()

      ctx.restore()
    })
    //    Draw the bubble background.
    ctx.globalAlpha = 0.7
    ctx.beginPath()
    ChartUtils.roundRect(ctx, bubbleBox, 10)
    ctx.fillStyle = ChartUtils.color.gray[1000]
    ctx.fill()
    ctx.strokeStyle = ChartUtils.color.gray[1000]
    ctx.stroke()
    highlightIndexes.forEach((highlightI, i) => {
      const box = {...boxWithoutY, y: fGet(this._boxYsInfo)[i].curr}

      ctx.save()
      ctx.globalAlpha = 0.8

      // Draw the bubble background.
      // ctx.beginPath()
      // ChartUtils.roundRect(ctx, box, 5)
      // ctx.fillStyle = ChartUtils.color.gray[1000]
      // ctx.fill()

      // Draw the text.
      ctx.globalAlpha = 1
      ctx.fillStyle = ChartUtils.color.gray[200]
      // the -1.5 is for visual center.
      const textY = box.y - 1 + box.height - padding.y / 2

      ctx.textAlign = 'right'
      ctx.fillText(
        texts[i].left,
        box.x + padding.x / 2 + maxWidthMeasureLeft.width,
        textY
      )
      // ctx.textAlign = 'left'
      // ctx.fillText(
      //   texts[i].middle,
      //   box.x + padding.x / 2 + maxWidthMeasureLeft.width + textGapX,
      //   textY
      // )
      ctx.textAlign = 'right'
      ctx.fillText(texts[i].right, box.x + box.width - padding.x / 2, textY)

      ctx.restore()
    })
    ctx.restore()
  }
}
