import { letIn, noCase } from '@tpaw/common'
import _ from 'lodash'
import { Rect } from '../../../../Utils/Geometry'
import { interpolate } from '../../../../Utils/Interpolate'
import { transitionTransform } from '../../../../Utils/Transition'
import { PlanColors } from '../../../PlanRoot/Plan/UsePlanColors'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import {
  ChartComponent,
  ChartComponentPointerTargetY,
  ChartRegisterAnimation,
} from './ChartComponent'
import { chartMergeDataComponentsArrTransition } from './ChartMergeDataComponentsArrTransition'

const vertGapToPoint = 5
const maxVertCollisionAvoidance = 30
const overlapDetectionPadding = 5
export class ChartSalientPoints<Params> implements ChartComponent<Params> {
  constructor(
    private _propsFn: (params: Params) => {
      salientPoints: {
        id: string
        dataX: number
        dataY: number
        label: string
        opacity: number
        align: 'above' | 'below'
        collisionDataId: string
      }[]
      colors: PlanColors['results']
      fontSize: number
    },
  ) {}

  public draw(
    context: ChartContext<Params>,
    reason:
      | 'draw'
      | 'init'
      | 'pointer'
      | 'stateAndPointer'
      | 'sizingAndPointer',
    registerAnimation: ChartRegisterAnimation,
    pointerTargetY: ChartComponentPointerTargetY,
  ) {
    const { canvasContext: ctx, stateTransition } = context
    const propsTransition = transitionTransform(stateTransition, (x) =>
      this._propsFn(x.params),
    )
    if (
      propsTransition.from.salientPoints.length === 0 &&
      propsTransition.target.salientPoints.length === 0
    )
      return null

    const salientPointsTransition = letIn(
      _.unzip(
        chartMergeDataComponentsArrTransition(
          propsTransition.from.salientPoints,
          propsTransition.target.salientPoints,
        )
          .map(({ from, target }) =>
            !from
              ? { from: { ...target, opacity: 0 }, target }
              : !target
              ? { from, target: { ...from, opacity: 0 } }
              : { from, target },
          )
          .map((x) => [x.from, x.target]),
      ),
      ([from, target]) => ({ from, target }),
    )

    const fullTransition = transitionTransform(
      stateTransition,
      (x, ___, src) => ({
        ...x,
        props: {
          ...propsTransition[src],
          salientPoints: salientPointsTransition[src],
        },
      }),
    )

    ctx.textBaseline = 'bottom'

    const pixelSpec = interpolate(
      transitionTransform(
        fullTransition,
        ({ props, derivedState }, __, src) => {
          const { scale, plotArea } = derivedState
          const { salientPoints, colors, fontSize } = props

          ctx.font = ChartUtils.getMonoFont(fontSize)
          const preCollision = salientPoints.map(
            ({ label, dataX, dataY, align, opacity, collisionDataId }) => {
              const measure = ctx.measureText(label)
              const width = measure.width
              const height = measure.actualBoundingBoxAscent // Descent not needed because baseline is bottom.
              const x = _.clamp(
                scale.x.start(dataX) - width / 2,
                plotArea.x,
                plotArea.right - width,
              )

              const dataLinePixelYs = _.compact(
                [..._.range(x, x + width, 5), x + width].map(
                  (pixelX) =>
                    pointerTargetY(scale.x.inverse.notRounded(pixelX)).find(
                      ({ id }) => id === collisionDataId,
                    )?.pixelYTransition[src] ?? null,
                ),
              )
              const y = _moveAwayFromDataLIne(
                dataLinePixelYs,
                align,
                scale.y(dataY),
                height,
                plotArea,
              )

              return {
                width,
                height,
                x,
                y,
                label,
                opacity,
                fontSize,
              }
            },
          )
          const result = preCollision.map((curr, i) =>
            preCollision
              .slice(0, i)
              .some((prev) => prev.opacity !== 0 && _doesOverlap(prev, curr))
              ? { ...curr, opacity: 0 }
              : curr,
          )
          return result
        },
      ),
    )

    pixelSpec.forEach((spec) =>
      ChartUtils.saveRestore(ctx, () => {
        ctx.globalAlpha = spec.opacity
        ctx.fillStyle = fullTransition.target.props.colors.fg
        ctx.fillText(spec.label, spec.x, spec.y)
      }),
    )
    return null
  }
}

const _moveAwayFromDataLIne = (
  ys: number[],
  vertAlign: 'above' | 'below',
  targetPixelY: number,
  height: number,
  plotArea: Rect,
) => {
  const applyBounds = (pixelY: number) =>
    Math.min(pixelY, plotArea.y + plotArea.height)
  if (vertAlign === 'above') {
    const originalY = targetPixelY - vertGapToPoint
    const collisionFreeY = Math.min(...ys, originalY)
    return applyBounds(
      collisionFreeY < originalY - maxVertCollisionAvoidance
        ? originalY
        : collisionFreeY,
    )
  }

  if (vertAlign === 'below') {
    const originalY = targetPixelY + vertGapToPoint
    const collisionFreeY = Math.max(...ys, originalY)
    return applyBounds(
      (collisionFreeY > originalY + maxVertCollisionAvoidance
        ? originalY
        : collisionFreeY) + height,
    )
  }
  noCase(vertAlign)
}

const _doesOverlap = (a: Rect, b: Rect) => {
  const left = (a: Rect) => a.x - overlapDetectionPadding
  const right = (a: Rect) => a.x + a.width + overlapDetectionPadding
  const top = (a: Rect) => a.y - overlapDetectionPadding
  const bottom = (a: Rect) => a.y + a.height + overlapDetectionPadding
  // no horizontal overlap
  if (left(a) >= right(b) || left(b) >= right(a)) return false
  // no vertical overlap
  if (top(a) >= bottom(b) || top(b) >= bottom(a)) return false
  return true
}
