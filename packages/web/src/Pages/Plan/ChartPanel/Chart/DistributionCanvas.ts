import {localPoint} from '@visx/event'
import {gsap} from 'gsap'
import _ from 'lodash'
import { UseTPAWWorkerResult } from '../../../../TPAWSimulator/Worker/UseTPAWWorker'
import {
  linearFnFomPoints,
  linearFnFromPointAndSlope
} from '../../../../Utils/LinearFn'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { fGet } from '../../../../Utils/Utils'
import { ChartContext } from './ChartContext'
import { ChartPointer } from './ChartPointer'
import { Size } from './ChartUtils'
import { drawAgeAxis } from './DrawAgeAxis'
import { drawPercentiles } from './DrawPercentiles'
import { drawSmallYAxis } from './DrawSmallYAxis'


type _Size = {width: number; height: number}
export class DistributionCanvas {
  private _canvas
  private _ctx: CanvasRenderingContext2D
  private _size: _Size

  private _tpawResult
  private _maxY: {target: number; curr: number}
  private _dataAnimation: ReturnType<typeof gsap['timeline']> | null
  private _yAxisAnimation: ReturnType<typeof gsap['timeline']> | null
  private _percentileInfo: {
    highlightIndexes: number[]
    backgroundIndexes: number[]
  }
  private _pointer: ChartPointer
  constructor(
    canvas: HTMLCanvasElement,
    size: _Size,
    tpawResult: UseTPAWWorkerResult,
    highlightPercentiles: number[],
    maxY: number
  ) {
    this._canvas = canvas
    this._canvas.onpointermove = e => {
      this._pointer.setPosition(fGet(localPoint(this._canvas, e)))
      this._draw()
    }
    this._canvas.onpointerenter = () => this._pointer.show()
    this._canvas.onpointerleave = () => this._pointer.hide()

    this._size = size
    this._ctx = fGet(canvas.getContext('2d'))
    const dpr = window.devicePixelRatio || 1
    this._ctx.scale(dpr, dpr)
    this._tpawResult = _processTPAWResult(tpawResult, null)
    this._maxY = {target: maxY, curr: maxY}
    this._setSizeImpl(size)

    this._dataAnimation = null
    this._yAxisAnimation = null
    const highlightIndexes = highlightPercentiles.map(x =>
      tpawResult.args.percentiles.indexOf(x)
    )
    this._pointer = new ChartPointer(() => this._draw())
    // this._pointer.show()

    this._percentileInfo = {
      highlightIndexes,
      backgroundIndexes: _.difference(
        _.range(tpawResult.args.percentiles.length),
        highlightIndexes
      ),
    }
  }
  setData(tpawResult: UseTPAWWorkerResult) {
    if (this._dataAnimation) {
      this._dataAnimation.kill()
    }
    this._tpawResult = _processTPAWResult(tpawResult, this._tpawResult)
    this._dataAnimation = gsap.timeline({
      onUpdate: () => this._draw(),
      ease: 'power4',
    })
    this._dataAnimation.to(
      this._tpawResult.age,
      {
        start: this._tpawResult.targetAge.start,
        end: this._tpawResult.targetAge.end,
        duration: 0.5,
      },
      0
    )
    this._dataAnimation.to(
      this._tpawResult,
      {dataTransition: 1, duration: 0.5},
      0
    )
  }
  setMaxY(maxY: number) {
    this._maxY = {target: maxY, curr: this._maxY.curr}
    this._yAxisAnimation = gsap.timeline({
      onUpdate: () => this._draw(),
      ease: 'power4',
    })
    this._yAxisAnimation.to(this._maxY, {
      curr: this._maxY.target,
      duration: 0.5,
    })
  }
  private _draw() {
    const tpawResult = this._tpawResult
    const size = this._size
    const ctx = this._ctx
    const chartContext = _chartContext(
      ctx,
      tpawResult,
      this._percentileInfo,
      this._maxY,
      size
    )
    const {plotArea} = chartContext

    ctx.clearRect(0, 0, this._size.width, this._size.height)
    ctx.lineWidth = 1

    // ctx.beginPath()
    // ChartUtils.roundRect(ctx, plotArea, 10)
    // ctx.fillStyle = ChartUtils.color.gray['100']
    // ctx.fill()

    drawAgeAxis(chartContext)
    drawPercentiles(chartContext)
    drawSmallYAxis(chartContext)
    // drawHeading(chartContext)
    this._pointer?.draw(chartContext)
  }

  setSize(size: _Size) {
    this._setSizeImpl(size)
    this._draw()
  }

  private _setSizeImpl(size: _Size) {
    const canvas = this._canvas
    canvas.style.width = `${size.width}px`
    canvas.style.height = `${size.height}px`
    const dpr = window.devicePixelRatio || 1
    canvas.width = size.width * dpr
    canvas.height = size.height * dpr
    this._ctx.scale(dpr, dpr)
    this._size = size
  }
}

const _chartContext = (
  ctx: CanvasRenderingContext2D,
  tpawResult: UseTPAWWorkerResultProcessed,
  percentileInfo: ChartContext['percentileInfo'],
  maxY: {target: number; curr: number},
  size: Size
): ChartContext => {
  // Padding on top to allow for tooltip bleeding out of plotArea.
  const padding = {
    left: 10,
    top: Math.max(30, size.height - size.width * 0.65),
    bottom: 35,
    right: 10,
  }
  const viewPort = {x: 0, y: 0, ...size}
  const headingHeight = 0
  const plotArea = {
    x: padding.left,
    y: padding.top,
    width: viewPort.width - padding.left - padding.right,
    height: viewPort.height - padding.top - padding.bottom,
  }

  const getScale = (x: SimpleRange, ySize: number) => ({
    x: linearFnFomPoints(
      x.start,
      plotArea.x,
      x.end,
      plotArea.width + plotArea.x
    ),
    y: linearFnFomPoints(
      0,
      plotArea.y + plotArea.height,
      ySize,
      plotArea.y + headingHeight
    ),
  })
  return {
    ctx,
    tpawResult,
    percentileInfo,
    maxY,
    padding,
    viewPort,
    plotArea,
    scale: {
      curr: getScale(tpawResult.age, maxY.curr),
      target: getScale(tpawResult.targetAge, maxY.target),
    },
  }
}

export type UseTPAWWorkerResultProcessed = {
  src: UseTPAWWorkerResult
  data: ((age: number) => number)[]
  prevData: UseTPAWWorkerResultProcessed['data']
  dataTransition: number
  targetAge: {start: number; end: number}
  age: {start: number; end: number}
}
const _processTPAWResult = (
  src: UseTPAWWorkerResult,
  prev: UseTPAWWorkerResultProcessed | null
): UseTPAWWorkerResultProcessed => {
  const targetAge = _age(src)
  const age = prev ? prev.age : targetAge
  const beforeSlope = _avgSlope(
    src.withdrawalsByPercentileByYearsIntoRetirement,
    0
  )
  const afterSlope = _avgSlope(
    src.withdrawalsByPercentileByYearsIntoRetirement,
    src.withdrawalsByPercentileByYearsIntoRetirement[0].data.length - 1
  )
  const data = src.withdrawalsByPercentileByYearsIntoRetirement.map(
    ({data: withdrawalsByYearsIntoRetirement}) => {
      const extrapolateBefore = linearFnFromPointAndSlope(
        targetAge.start,
        withdrawalsByYearsIntoRetirement[0],
        beforeSlope
      )
      const extrapolateAfter = linearFnFromPointAndSlope(
        targetAge.end,
        fGet(_.last(withdrawalsByYearsIntoRetirement)),
        afterSlope
      )
      const interpolated = withdrawalsByYearsIntoRetirement
        .slice(0, -1)
        .map((v, i) =>
          linearFnFomPoints(
            i + targetAge.start,
            v,
            i + targetAge.start + 1,
            withdrawalsByYearsIntoRetirement[i + 1]
          )
        )
      return (a: number) =>
        a <= targetAge.start
          ? extrapolateBefore(a)
          : a < targetAge.end
          ? interpolated[Math.floor(a) - targetAge.start](a)
          : extrapolateAfter(a)
    }
  )
  return {
    src,
    data,
    prevData: prev?.data ?? data,
    dataTransition: 0,
    targetAge,
    age,
  }
}

const _age = (result: UseTPAWWorkerResult) => {
  const {params} = result.args
  const start = params.age.retirement
  const end = params.age.end
  return {start, end}
}

const _avgSlope = (byPercentileByYears: {data: number[]}[], i: number) => {
  const iPlus1 = Math.min(i + 1, byPercentileByYears[0].data.length - 1)
  return (
    _.sum(byPercentileByYears.map(x => x.data[iPlus1] - x.data[i])) /
    byPercentileByYears.length
  )
}
