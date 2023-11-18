import { assert, block, fGet, letIn, noCase } from '@tpaw/common'
import { Power1, gsap } from 'gsap'
import _ from 'lodash'
import { RGB } from '../../../../Utils/ColorUtils'
import { Rect, XY } from '../../../../Utils/Geometry'
import { interpolate } from '../../../../Utils/Interpolate'
import { Transition, transitionTransform } from '../../../../Utils/Transition'
import { ChartAnimatedProperty, ChartAnimation } from '../Chart'
import { ChartContext } from '../ChartContext'
import { ChartStyling } from '../ChartUtils/ChartStyling'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import {
  ChartComponent,
  ChartComponentPointerTargetY,
  ChartRegisterAnimation,
} from './ChartComponent'
import { ChartPointerPortal } from './ChartPointerPortal'

const _localSlowAnimation: ChartAnimation = {
  ease: Power1.easeOut,
  duration: 500 / 1000,
}

export type ChartPointerProps = {
  getBox: (
    dataX: number,
    hoverTransitionMap: Map<string, Transition<0 | 1> | null>,
    components: ChartPointerBoxComponents,
  ) => ReturnType<ChartPointerBoxComponents['box']>
  positioning: (args: { width: number; height: number; hover: number }) => {
    origin: XY
    clip: Rect
  }
}

export class ChartPointer<Params> implements ChartComponent<Params> {
  private _animatedProps: ChartAnimatedProperty<{
    width: number
    height: number
  }> | null = null
  constructor(
    private readonly _portal: ChartPointerPortal,
    private readonly _getProps: (
      params: ChartContext<Params>['stateTransition']['from'],
    ) => ChartPointerProps,
  ) {}

  public destroy() {
    this._animatedProps?.animation?.kill()
  }

  private _update(
    context: ChartContext<Params>,
    change: 'init' | 'pointer' | 'stateAndPointer' | 'sizingAndPointer',
    registerAnimation: ChartRegisterAnimation,
    pointerTargetY: ChartComponentPointerTargetY,
  ) {
    const { stateTransition, pointerTransition } = context

    const target = block(() => {
      const { scale } = stateTransition.target.derivedState
      const pixelXAtTarget = pointerTransition.visualPosition.target.x
      const pointerDataX = scale.x.inverse.rounded(pixelXAtTarget)
      const { getBox } = this._getProps(stateTransition.target)

      const ysUnfiltered = pointerTargetY(pointerDataX)
      const hoverTransitionMap = new Map(
        ysUnfiltered.map(({ id, hoverTransition }) => [id, hoverTransition]),
      )
      const box = getBox(
        pointerDataX,
        hoverTransitionMap,
        _getChartPointerBoxComponents(fGet(this._portal.data).ctx),
      )()

      return {
        width: box.width,
        height: box.height,
      }
    })
    if (!_.isEqual(target, this._animatedProps?.transition?.target)) {
      const from = this._animatedProps
        ? interpolate(this._animatedProps.transition)
        : target
      this._animatedProps?.animation?.kill()
      this._animatedProps = {
        transition: { from, target, progress: 0 },
        animation: null,
      }
      if (change === 'sizingAndPointer') {
        this._animatedProps.transition.progress = 1
      } else {
        this._animatedProps.animation = registerAnimation(
          gsap.fromTo(
            this._animatedProps.transition,
            { progress: 0 },
            {
              progress: 1,
              ease: _localSlowAnimation.ease,
              duration: _localSlowAnimation.duration,
            },
          ),
        )
      }
    }
  }

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
    if (!this._portal.data) return null

    if (reason !== 'draw') {
      this._update(context, reason, registerAnimation, pointerTargetY)
    }

    const ctx = this._portal.data.ctx
    const portalViewport = this._portal.data.viewport
    ctx.clearRect(
      portalViewport.x,
      portalViewport.y,
      portalViewport.width,
      portalViewport.height,
    )
    const { stateTransition, currState, pointerTransition } = context
    const fullTransition = transitionTransform(stateTransition, (x) => ({
      ...x,
      props: this._getProps(x),
    }))
    const animatedProps = interpolate(fGet(this._animatedProps).transition)

    // Draw the box.
    {
      const width = animatedProps.width
      const height = animatedProps.height
      const { positioning } = fullTransition.target.props
      const { origin, clip } = positioning({
        width,
        height,
        hover: currState.pointer.hover,
      })
      const boxPosition = {
        width,
        height,
        ...origin,
      }

      const box = block(() => {
        const pointerDataX =
          fullTransition.target.derivedState.scale.x.inverse.rounded(
            pointerTransition.visualPosition.target.x,
          )
        return fullTransition.target.props.getBox(
          pointerDataX,
          new Map(
            pointerTargetY(pointerDataX).map(({ id, hoverTransition }) => [
              id,
              hoverTransition,
            ]),
          ),
          _getChartPointerBoxComponents(ctx),
        )()
      })

      ChartUtils.saveRestore(ctx, () => {
        ctx.clip(
          block(() => {
            const path = new Path2D()
            path.rect(clip.x, clip.y, clip.width, clip.height)
            return path
          }),
        )
        box.draw(boxPosition)
      })
    }

    // // Draw the dots.
    // ChartUtils.saveRestore(ctx, () => {
    //   ctx.globalAlpha = 0.7
    //   pixelComponents.pixelYInfos.forEach(({ pixelY }) => {
    //     ctx.fillStyle = colors.darkBG
    //     ctx.beginPath()
    //     ctx.ellipse(pointerInPixelCoord.x, pixelY, 5, 5, 0, 0, Math.PI * 4)
    //     ctx.fill()
    //   })
    // })

    // // Draw the lines.
    // ChartUtils.saveRestore(ctx, () => {

    //   ctx.save()
    //   ctx.lineWidth = 1
    //   ctx.strokeStyle = colors.darkBG

    //   pixelComponents.pixelYInfos
    //     // Because if it goes to far out, the lines won't draw.
    //     .map(({ id, pixelY }) => ({ id, pixelY: Math.max(pixelY, -10000000) }))
    //     .forEach(({ id, pixelY }) => {
    //       const pixelYInBoxUnclamped = boxDataPixelPositionsById.get(id)
    //       if (pixelYInBoxUnclamped === undefined) return
    //       const pixelYInBox = _.clamp(
    //         pixelYInBoxUnclamped,
    //         boxPosition.y + 10,
    //         boxPosition.y + boxPosition.height - 10,
    //       )

    //       ctx.beginPath()
    //       const line = [
    //         { x: pointerPosition.visual.x, y: pixelY },
    //         {
    //           x:
    //             pointerPosition.visual.x +
    //             (xLineTarget - pointerPosition.visual.x) * 0.6,
    //           y: pixelYInBox,
    //         },
    //         { x: xLineTarget, y: pixelYInBox },
    //       ]
    //       ChartUtils.roundedLine(ctx, line, 10)
    //       ctx.stroke()
    //       // Not sure why this is necessary, but without it, when there is only
    //       // one line, it seemed to fill the path with white.
    //       ctx.beginPath()
    //     })
    // })
    return null
  }
}

// ------------------------
// ---- BOX COMPONENTS ----
// ------------------------

export type ChartPointerBoxComponents = ReturnType<
  typeof _getChartPointerBoxComponents
>

const _getChartPointerBoxComponents = (ctx: CanvasRenderingContext2D) => {
  const _mergeDrawResults = (x: Map<string, number>[]) =>
    new Map(_.flatten(_.compact(x).map((x) => [...x])))
  type Padding = {
    left?: number
    right?: number
    top?: number
    bottom?: number
  }
  const _applyDefaultToPadding = (padding?: Padding) => ({
    left: padding?.left ?? 0,
    right: padding?.right ?? 0,
    top: padding?.top ?? 0,
    bottom: padding?.bottom ?? 0,
  })

  const noOp = { width: 0, height: 0, ids: [], draw: () => new Map() }

  type TextNode = ReturnType<typeof text>
  const text =
    (
      opts: {
        maxWidth?: number
        yOffset?: number
        padding?: { left?: number; right?: number }
      },
      textIn: string,
    ) =>
    // This level of indirection is needed for styles to be applied in the right
    // order for size calculations.
    () => {
      const { maxWidth } = opts
      const padding = letIn(
        _applyDefaultToPadding(opts.padding),
        ({ left, right }) => ({ left, right }),
      )
      const getPaddedTextWidth = ({ width }: { width: number }) =>
        width + padding.left + padding.right
      const textInMeasure = ctx.measureText(textIn)
      const textInfo =
        maxWidth !== undefined && getPaddedTextWidth(textInMeasure) > maxWidth
          ? block((): { text: string; measure: typeof textInMeasure } => {
              const removeChars = (numCharsToRemove: number) => {
                const nPre = Math.ceil((textIn.length - numCharsToRemove) / 2)
                const nPost = Math.floor((textIn.length - numCharsToRemove) / 2)
                const pre = textIn.slice(0, nPre).trimEnd()
                const post = textIn.slice(textIn.length - nPost).trimStart()
                const text = `${pre} … ${post}`
                const measure = ctx.measureText(text)
                return getPaddedTextWidth(measure) < maxWidth
                  ? { text: text, measure }
                  : null
              }
              const approxPixelsPerChar = textInMeasure.width / textIn.length
              const approxNumCharsToRemove = Math.ceil(
                (getPaddedTextWidth(textInMeasure) - maxWidth) /
                  approxPixelsPerChar,
              )
              const startingInfo = removeChars(approxNumCharsToRemove)
              if (startingInfo === null) {
                for (const n of _.range(
                  approxNumCharsToRemove + 1,
                  textIn.length + 1,
                )) {
                  const info = removeChars(n)
                  if (info) return info
                }
                return { text: '', measure: ctx.measureText('') }
              } else {
                let prev = startingInfo
                for (const n of _.range(approxNumCharsToRemove - 1, -1, -1)) {
                  const info = removeChars(n)
                  if (!info) return prev
                  prev = info
                }
                assert(false)
              }
            })
          : { text: textIn, measure: textInMeasure }

      const { actualBoundingBoxAscent, actualBoundingBoxDescent } =
        textInfo.measure

      // To support firebox < 116
      const fontBoundingBoxAscent =
        textInfo.measure.fontBoundingBoxAscent ??
        textInfo.measure.actualBoundingBoxAscent
      const fontBoundingBoxDescent =
        textInfo.measure.fontBoundingBoxDescent ??
        textInfo.measure.actualBoundingBoxDescent

      const yOffset = opts.yOffset ?? 0

      return {
        width: getPaddedTextWidth(textInfo.measure),
        actualBoundingBoxAscent: actualBoundingBoxAscent + yOffset,
        actualBoundingBoxDescent: actualBoundingBoxDescent - yOffset,
        fontBoundingBoxAscent: fontBoundingBoxAscent + yOffset,
        fontBoundingBoxDescent: fontBoundingBoxDescent - yOffset,
        draw: ({ x, y }: XY) =>
          ChartUtils.saveRestore(ctx, () => {
            ctx.textBaseline = 'alphabetic'
            ctx.textAlign = 'start'
            ctx.fillText(textInfo.text, x + padding.left, y - yOffset)
          }),
      }
    }

  type SectionNode = () => {
    width: number
    height: number
    ids: string[]
    draw: (xy: XY, width: number) => Map<string, number>
  }

  const getLabel =
    ({ lineHeight }: { lineHeight: number }) =>
    (
      opts: {
        align?: 'start' | 'end' | 'center'
        id?: string
        underline?: { gap: number; height: number; percent?: number }
        box?: {
          borderRadius?: number
          padding?: Padding
          fill?: { color: RGB; opacity?: number }
          stroke?: ChartStyling.Stroke
          percent?: number
        }
      },
      partsIn: TextNode[] | TextNode,
    ): SectionNode =>
    () => {
      const align = opts.align ?? 'start'

      const parts = _.flatten([partsIn]).map((x) => x())
      const fontBoundingBoxAscent = Math.max(
        ...parts.map((x) => x.fontBoundingBoxAscent),
      )
      const fontBoundingBoxDescent = Math.max(
        ...parts.map((x) => x.fontBoundingBoxDescent),
      )
      const actualWidth = _.sum(parts.map((x) => x.width))
      return {
        width: actualWidth,
        ids: opts.id ? [opts.id] : [],
        height: lineHeight,
        draw: (xy: XY, width: number) => {
          ChartUtils.saveRestore(ctx, () => {
            if (opts.box) {
              const padding = _applyDefaultToPadding(opts.box.padding)
              const rect = {
                x: xy.x - padding.left,
                y: xy.y - padding.top,
                width:
                  (actualWidth + padding.left + padding.right) *
                  (opts.box.percent ?? 1),
                height: lineHeight + padding.top + padding.bottom,
              }
              const path = ChartUtils.roundRect(
                rect,
                opts.box.borderRadius ?? 0,
              )
              if (opts.box.fill) {
                ctx.globalAlpha = opts.box.fill.opacity ?? 1
                ctx.fillStyle = RGB.toHex(opts.box.fill.color)
                ctx.fill(path)
              }
              if (opts.box.stroke && rect.width > 0) {
                ctx.globalAlpha = opts.box.stroke.opacity
                ctx.strokeStyle = RGB.toHex(opts.box.stroke.color)
                ctx.lineWidth = opts.box.stroke.width * (opts.box.percent ?? 1)
                ctx.stroke(path)
              }
            }
          })

          return ChartUtils.saveRestore(ctx, () => {
            const bottom = xy.y + lineHeight
            const baseline = bottom - fontBoundingBoxDescent
            const textMiddle =
              bottom - (fontBoundingBoxAscent + fontBoundingBoxDescent) / 2
            const startX =
              align === 'start'
                ? xy.x
                : align === 'end'
                ? xy.x + width - actualWidth
                : align === 'center'
                ? xy.x + (width - actualWidth) / 2
                : noCase(align)
            let x = startX
            parts.forEach((textInfo) => {
              textInfo.draw({ x, y: baseline })
              x += textInfo.width
            })

            ChartUtils.saveRestore(ctx, () => {
              if (opts.underline) {
                const path = new Path2D()
                const left = align === 'start' ? startX : x
                const width = Math.abs(x - startX)
                const y = baseline + opts.underline.gap
                path.moveTo(left, y)
                path.lineTo(left + width * (opts.underline.percent ?? 1), y)
                ctx.lineWidth = opts.underline.height
                ctx.stroke(path)
              }
            })

            return new Map(opts.id ? [[opts.id, textMiddle]] : [])
          })
        },
      }
    }

  const pad = (paddingIn: Padding, sectionIn: SectionNode) => () => {
    const padding = _applyDefaultToPadding(paddingIn)
    const section = sectionIn()

    return {
      width: padding.left + section.width + padding.right,
      height: padding.top + section.height + padding.bottom,
      ids: section.ids,
      draw: (xy: XY, width: number) =>
        section.draw(
          { x: xy.x + padding.left, y: xy.y + padding.top },
          width - padding.left - padding.right,
        ),
    }
  }
  const mark = (sectionIn: SectionNode) => () => {
    const section = sectionIn()

    return {
      width: section.width,
      height: section.height,
      ids: section.ids,
      draw: (xy: XY, width: number) => {
        ChartUtils.saveRestore(ctx, () => {
          ctx.fillStyle = 'red'
          ctx.fillRect(xy.x, xy.y, width, section.height)
        })
        return section.draw(xy, width)
      },
    }
  }
  const fixedWidth =
    (
      { width, align }: { width: number; align: 'start' | 'end' | 'center' },
      sectionIn: SectionNode,
    ) =>
    () => {
      const section = sectionIn()

      return {
        width,
        height: section.height,
        ids: section.ids,
        draw: (xy: XY, width: number) => {
          const x =
            align === 'start'
              ? xy.x
              : align === 'end'
              ? xy.x + width - section.width
              : align === 'center'
              ? xy.x + (width - section.width) / 2
              : noCase(align)
          return section.draw({ x: x, y: xy.y }, width)
        },
      }
    }
  const grid =
    (
      opts: {
        gap: { horz: number; vert: number }
        align: {
          horz: 'start' | 'end' | 'between'
          vert: 'start' | 'end' | 'center'
        }
      },
      rowsIn: SectionNode[][],
    ) =>
    () => {
      if (rowsIn.length === 0) return noOp
      const rows = rowsIn.map((colsIn) => colsIn.map((x) => x()))
      const numColsByRow = rows.map((x) => x.length)
      assert(_.uniq(numColsByRow).length === 1)
      const numCols = numColsByRow[0]
      if (numCols === 0) return noOp

      const rowHeights = rows.map((cols) =>
        fGet(_.max(cols.map((x) => x.height))),
      )
      const colWidths = _.range(numCols).map((i) =>
        fGet(_.max(rows.map((x) => x[i].width))),
      )

      const minWidth = _.sum(colWidths) + (numCols - 1) * opts.gap.horz
      return {
        width: minWidth,
        height: _.sum(rowHeights) + (rows.length - 1) * opts.gap.vert,
        ids: _.flatten(rows.map((col) => _.flatten(col.map((x) => x.ids)))),
        draw: (xy: XY, width: number) => {
          const extraW = width - minWidth
          const xs = block(() => {
            const xs = [] as number[]
            let x = xy.x
            switch (opts.align.horz) {
              case 'start':
                colWidths.forEach((w) => {
                  xs.push(x)
                  x = x + w + opts.gap.horz
                })
              case 'end':
                x += extraW
                colWidths.forEach((w) => {
                  xs.push(x)
                  x = x + w + opts.gap.horz
                })
              case 'between':
                const gap = opts.gap.horz + extraW / (numCols - 1)
                colWidths.forEach((w) => {
                  xs.push(x)
                  x = x + w + gap
                })
            }
            return xs
          })
          return _mergeDrawResults(
            rows.map((cols, ri) => {
              const rowHeight = rowHeights[ri]
              return _mergeDrawResults(
                cols.map((item, ci) => {
                  const yTop = xy.y + ri * (rowHeight + opts.gap.vert)
                  const y =
                    opts.align.vert === 'start'
                      ? yTop
                      : opts.align.vert === 'center'
                      ? yTop + (rowHeight - item.height) / 2
                      : opts.align.vert === 'end'
                      ? yTop + (rowHeight - item.height)
                      : noCase(opts.align.vert)
                  return item.draw({ x: xs[ci], y }, colWidths[ci])
                }),
              )
            }),
          )
        },
      }
    }

  const oneRowGrid = (
    opts: {
      gap: number
      align: {
        horz: 'start' | 'end' | 'between'
        vert: 'start' | 'end' | 'center'
      }
    },
    cols: SectionNode[],
  ) =>
    grid(
      {
        gap: { horz: opts.gap, vert: 0 },
        align: opts.align,
      },
      [cols],
    )

  const roundedRect =
    (optsIn: { radius: number; width: number; height: number; id?: string }) =>
    () => {
      const opts = {
        ...optsIn,
      }
      return {
        width: opts.width,
        height: opts.height,
        ids: opts.id ? [opts.id] : [],
        draw: (xy: XY) => {
          const path = ChartUtils.roundRect(
            { ...xy, width: opts.width, height: opts.height },
            opts.radius,
          )
          ctx.fill(path)
          return new Map(opts.id ? [[opts.id, xy.y + opts.radius]] : [])
        },
      }
    }

  const circle =
    (opts: {
      box: { width: number; height: number }
      center: XY
      radius: number
      id?: string
    }) =>
    () => {
      return {
        width: opts.box.width,
        height: opts.box.height,
        ids: opts.id ? [opts.id] : [],
        draw: (xy: XY) => {
          const path = new Path2D()
          path.ellipse(
            xy.x + opts.center.x,
            xy.y + opts.center.y,
            opts.radius,
            opts.radius,
            0,
            0,
            Math.PI * 2,
          )
          ctx.fill(path)
          return new Map(opts.id ? [[opts.id, xy.y + opts.radius]] : [])
        },
      }
    }

  const gap =
    (height: number, width = 0): SectionNode =>
    () => {
      return {
        width,
        height: height,
        ids: [],
        draw: () => new Map(),
      }
    }

  const hr =
    (lineHeight: number): SectionNode =>
    () => ({
      width: 0,
      height: lineHeight,
      ids: [],
      draw: (xy: XY, width: number) => {
        const path = new Path2D()
        const y = xy.y + lineHeight / 2
        path.moveTo(xy.x, y)
        path.lineTo(xy.x + width, y)
        ctx.stroke(path)
        return new Map()
      },
    })

  const barForText =
    ({
      text,
      height,
      align,
    }: {
      height: number
      text: string
      align: 'start' | 'end'
    }): SectionNode =>
    () => {
      const width = ctx.measureText(text).width
      return {
        width,
        height,
        ids: [],
        draw: (xy: XY, boxWidth: number) => {
          const y = xy.y + height / 2
          const path = new Path2D()
          const x = xy.x + (align === 'start' ? 0 : boxWidth - width)
          path.moveTo(x, y)
          path.lineTo(x + width, y)
          ctx.lineWidth = height
          ctx.stroke(path)
          return new Map()
        },
      }
    }

  const hoverBG =
    (height: number, horzPad: number, hover: number): SectionNode =>
    () => {
      return {
        width: 0,
        height: 0,
        ids: [],
        draw: (xy: XY, width: number) => {
          if (hover > 0) {
            const x = xy.x - horzPad
            const path = ChartUtils.roundRect(
              { x, y: xy.y, width: (width + horzPad * 2) * hover, height },
              5,
            )
            ctx.globalAlpha = hover
            ctx.fill(path)
          }
          return new Map()
        },
      }
    }
  const hoverBar =
    (height: number, hover: number): SectionNode =>
    () => {
      return {
        width: 0,
        height: 0,
        ids: [],
        draw: (xy: XY, width: number) => {
          const y = xy.y + height / 2
          const path = new Path2D()
          path.moveTo(xy.x, y)
          path.lineTo(xy.x + width * hover, y)
          ctx.lineWidth = height
          ctx.stroke(path)
          return new Map()
        },
      }
    }

  const group =
    (partsIn: SectionNode[]): SectionNode =>
    () => {
      const parts = _.flatten([partsIn]).map((x) => x())
      if (parts.length === 0) return noOp
      return {
        width: fGet(_.max(parts.map((x) => x.width))),
        height: _.sum(parts.map((x) => x.height)),
        ids: _.flatten(parts.map((x) => x.ids)),
        draw: (xy: XY, width: number) => {
          let y = xy.y
          return _mergeDrawResults(
            parts.map((section, i) =>
              ChartUtils.saveRestore(ctx, () => {
                const result = section.draw({ x: xy.x, y }, width)
                y += section.height
                return result
              }),
            ),
          )
        },
      }
    }

  type Style = {
    font?: string
    fillColor?: string
    strokeColor?: string
    opacity?: number
  }

  const applyStyle = ({ font, fillColor, strokeColor, opacity }: Style) => {
    if (font) ctx.font = font
    if (fillColor) ctx.fillStyle = fillColor
    if (strokeColor) ctx.strokeStyle = strokeColor
    if (opacity) ctx.globalAlpha = opacity
  }
  const style =
    <U extends any[], V, T extends { draw: (...args: U) => V }>(
      style: Style,
      tIn: () => T,
    ) =>
    () => {
      const t = ChartUtils.saveRestore(ctx, () => {
        applyStyle(style)
        return tIn()
      })
      return {
        ...t,
        draw: (...args: U) =>
          ChartUtils.saveRestore(ctx, () => {
            applyStyle(style)
            return t.draw(...args)
          }),
      }
    }

  const box =
    (
      opts: {
        padding: { vert: number; horz: number }
        borderRadius: number
        lineWidth: number
      },
      sectionsIn: SectionNode,
    ) =>
    () => {
      const section = sectionsIn()
      return {
        width: section.width + opts.padding.horz * 2,
        height: section.height + opts.padding.vert * 2,
        ids: section.ids,
        draw: (rect: Rect) =>
          ChartUtils.saveRestore(ctx, () => {
            const path = ChartUtils.roundRect(rect, opts.borderRadius)
            ctx.fill(path)
            if (opts.lineWidth > 0) {
              ChartUtils.saveRestore(ctx, () => {
                ctx.clip(path)
                ctx.lineWidth = opts.lineWidth * 2 // * 2 because half is clipped.
                ctx.stroke(path)
              })
            }
            const y = rect.y + opts.padding.vert
            const x = rect.x + opts.padding.horz
            ctx.clip(
              block(() => {
                const clipPath = new Path2D()
                clipPath.rect(
                  x,
                  y,
                  rect.width - opts.padding.horz,
                  rect.height - opts.padding.vert,
                )
                return clipPath
              }),
            )
            return section.draw({ x, y }, rect.width - opts.padding.horz * 2)
          }),
      }
    }

  return {
    text,
    group,
    getLabel,
    style,
    mark,
    pad,
    box,
    circle,
    grid,
    fixedWidth,
    oneRowGrid,
    hr,
    barForText,
    gap,
    hoverBar,
    roundedRect,
    hoverBG,
  }
}
