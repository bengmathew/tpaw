import _ from 'lodash'
import { Rect } from '../../../../Utils/Geometry'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { fGet } from '../../../../Utils/Utils'

export namespace ChartUtils {
  export const saveRestore = <T>(
    ctx: CanvasRenderingContext2D,
    fn: () => T,
  ) => {
    ctx.save()
    const result = fn()
    ctx.restore()
    return result
  }
  export type FontWeight = 'bold' | '400' | '500' | '600' | '800' | '900' | ''
  export const getFont = (size: number, style: FontWeight = '') =>
    `${style} ${size}px Montserrat, sans-serif`
  export const getMonoFont = (size: number, style: FontWeight = '') =>
    `${style} ${size}px Roboto Mono, monospace`

  export function roundRect({ x, y, width, height }: Rect, radius: number) {
    const right = x + width
    const bottom = y + height
    const edges = [
      { x, y },
      { x: right, y },
      { x: right, y: bottom },
      { x, y: bottom },
    ]
    return roundedPolygon(edges, radius)
  }

  // Thanks, https://stackoverflow.com/a/56214413
  type _Point = { x: number; y: number }
  export function roundedPolygon(points: _Point[], radius: number) {
    const path = new Path2D()
    const corners = _roundedPolygonCorners(points, radius)
    path.moveTo(corners[0].start.x, corners[0].start.y)
    for (const { start, ctrl, end } of corners) {
      path.lineTo(start.x, start.y)
      path.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    }
    path.closePath()
    return path
  }

  export const getRangePath = (
    pixelXYs: { x: number; yRange: SimpleRange }[],
  ) => {
    const path = new Path2D()
    path.moveTo(pixelXYs[0].x, pixelXYs[0].yRange.start)
    pixelXYs.forEach(({ x, yRange }) => {
      path.lineTo(x, yRange.start)
    })
    pixelXYs
      .slice()
      .reverse()
      .forEach(({ x, yRange }) => {
        path.lineTo(x, yRange.end)
      })
    path.closePath()
    return path
  }
  getRangePath.getBounds = (
    pixelXYs: { x: number; yRange: SimpleRange }[],
  ) => ({
    x: {
      start: fGet(_.first(pixelXYs)).x,
      end: fGet(_.last(pixelXYs)).x,
    },
    y: {
      start: Math.min(...pixelXYs.map((xy) => xy.yRange.end)),
      end: Math.max(...pixelXYs.map((xy) => xy.yRange.start)),
    },
  })

  export function roundedLine(
    ctx: CanvasRenderingContext2D,
    points: readonly _Point[],
    radius: number,
  ) {
    // const [last, ...corners] = _roundedPolygonCorners(points, radius)
    // ctx.moveTo(last[1].x, last[1].y)
    // for (const [start, ctrl, end] of corners.slice(0, -1)) {
    //   ctx.lineTo(start.x, start.y)
    //   ctx.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    // }
    // ctx.lineTo(points[0].x, points[0].y)

    const corners = _roundedPolygonCorners(points, radius)
    const first = fGet(corners.pop())
    const last = fGet(corners.pop())
    ctx.moveTo(first.ctrl.x, first.ctrl.y)
    for (const { start, ctrl, end } of corners) {
      ctx.lineTo(start.x, start.y)
      ctx.quadraticCurveTo(ctrl.x, ctrl.y, end.x, end.y)
    }
    ctx.lineTo(last.ctrl.x, last.ctrl.y)
  }

  function _roundedPolygonCorners(points: readonly _Point[], radius: number) {
    const distance = (p1: _Point, p2: _Point) =>
      Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    const lerp = (a: number, b: number, x: number) => a + (b - a) * x

    const lerp2D = (p1: _Point, p2: _Point, t: number) => ({
      x: lerp(p1.x, p2.x, t),
      y: lerp(p1.y, p2.y, t),
    })

    const numPoints = points.length

    const corners = []
    for (let i = 0; i < numPoints; i++) {
      const lastPoint = points[i]
      const thisPoint = points[(i + 1) % numPoints]
      const nextPoint = points[(i + 2) % numPoints]

      const lastEdgeLength = distance(lastPoint, thisPoint)
      const lastOffsetDistance = Math.min(lastEdgeLength / 2, radius)
      const start = lerp2D(
        thisPoint,
        lastPoint,
        lastOffsetDistance / lastEdgeLength,
      )

      const nextEdgeLength = distance(nextPoint, thisPoint)
      const nextOffsetDistance = Math.min(nextEdgeLength / 2, radius)
      const end = lerp2D(
        thisPoint,
        nextPoint,
        nextOffsetDistance / nextEdgeLength,
      )

      corners.push({ start, ctrl: thisPoint, end })
    }
    return corners
  }
}
