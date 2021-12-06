import {gsap} from 'gsap'
import _ from 'lodash'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {fGet} from '../../../Utils/Utils'
import {ChartContext} from './ChartContext'
import {ChartUtils} from './ChartUtils'

export class ChartPointer {
  private _position
  private _onDraw
  private _ageInfo: {target: number; curr: number} | null = null
  private _labelSide = {target: 1, curr: 1}
  private _transition = {curr: 0}
  private _pointAnimation: ReturnType<typeof gsap['to']> | null = null
  private _labelSideAnimation: ReturnType<typeof gsap['to']> | null = null
  private _transitionAnimation: ReturnType<typeof gsap['to']> | null = null

  constructor(onDraw: () => void) {
    this._position = {x: 0, y: 0}
    this._onDraw = onDraw
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

  destroy() {
    this._pointAnimation?.kill()
    this._labelSideAnimation?.kill()
  }

  draw(chartContext: ChartContext) {
    const {ctx, scale, tpawResult, percentileInfo, plotArea} = chartContext
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
      if (this._pointAnimation) this._pointAnimation.kill()
      this._ageInfo.target = targetAge
      this._pointAnimation = gsap.to(this._ageInfo, {
        curr: targetAge,
        duration: 1,
        onUpdate: this._onDraw,
        ease: 'power4',
      })
    }

    const age = this._ageInfo.curr
    const ageX = scale.curr.x(age)

    ctx.font = ChartUtils.getFont(12)
    ctx.textAlign = 'left'
    const texts = highlightIndexes.map(
      i =>
        `${tpawResult.src.args.percentiles[i]}th - ${formatCurrency(
          tpawResult.data[i](targetAge)
        )}`
    )
    const measure = fGet(
      _.maxBy(
        texts.map(text => ctx.measureText(text)),
        x => x.width
      )
    )
    const offsetX = 20
    const padding = {y: 7, x: 15}

    const ys = highlightIndexes.map((highlightI, i) => {
      const interpolatedY = linearFnFomPoints(
        0,
        tpawResult.prevData[highlightI](age),
        1,
        tpawResult.data[highlightI](age)
      )(tpawResult.dataTransition)
      // -2 because percentiles are offset.
      return scale.curr.y(interpolatedY) - 2
    })
    const textYs = ys.map(y => y - 20)
    const boxYs = textYs.map(
      y => y - measure.fontBoundingBoxAscent - padding.y / 2
    )
    const boxWithoutY = {
      x: ageX + offsetX,
      width: measure.width + padding.x,
      height:
        measure.fontBoundingBoxAscent +
        measure.fontBoundingBoxDescent +
        padding.y,
    }
    boxYs.reduceRight((a, y, i) => {
      if (i === boxYs.length - 1) return null
      boxYs[i] = Math.max(boxYs[i + 1] + boxWithoutY.height + 5, y)
      return null
    }, null)

    boxYs.forEach((y, i) => {
      const prevTop = i === 0 ? plotArea.y + plotArea.height : boxYs[i - 1]
      boxYs[i] = Math.min(prevTop - 5 - boxWithoutY.height, y)
    })

    const targetLabelSide =
      boxWithoutY.x + boxWithoutY.width + 20 > plotArea.x + plotArea.width
        ? -1
        : 1
    if (this._labelSide.target !== targetLabelSide) {
      if (this._labelSideAnimation) this._labelSideAnimation.kill()
      this._labelSide.target = targetLabelSide
      this._labelSideAnimation = gsap.to(this._labelSide, {
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
    )(this._labelSide.curr)

    const boxHandleX = linearFnFomPoints(
      1,
      boxWithoutY.x,
      -1,
      boxWithoutY.x + boxWithoutY.width
    )(this._labelSide.curr)

    highlightIndexes.forEach((highlightI, i) => {
      const y = ys[i]
      ctx.save()
      // ctx.globalAlpha = linearFnFomPoints(0, 0.5, 1, 1)(this._transition.curr) 
      ctx.globalAlpha = 1
      const scale = linearFnFomPoints(0, 1, 1, 1)(this._transition.curr)
      ctx.translate(ageX, y)
      ctx.scale(scale, scale)
      ctx.translate(-ageX, -y)

      ctx.beginPath()
      ctx.ellipse(ageX, y, 4, 4, 0, 0, Math.PI * 4)
      ctx.lineWidth = 2
      ctx.fillStyle = 'black'
      ctx.fill()

      const box = {...boxWithoutY, y: boxYs[i]}

      ctx.beginPath()
      ctx.moveTo(ageX, y)
      ctx.lineTo(boxHandleX, box.y + box.height * 0.5)
      ctx.lineWidth = 1
      ctx.strokeStyle = 'black'
      ctx.stroke()

      ctx.beginPath()
      ChartUtils.roundRect(ctx, box, 5)
      ctx.fillStyle = 'black'
      ctx.fill()

      // ctx.lineWidth = 1
      // ctx.strokeStyle = 'black'
      // ctx.stroke()

      ctx.globalAlpha = 1
      ctx.fillStyle = 'white'
      ctx.fillText(
        texts[i],
        box.x + padding.x / 2,
        boxYs[i] + measure.fontBoundingBoxAscent + padding.y / 2
      )
      ctx.restore()
    })

    ctx.restore()
  }
}
