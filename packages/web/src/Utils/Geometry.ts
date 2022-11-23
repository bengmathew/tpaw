import _ from 'lodash'

export type Rect = {
  x: number
  y: number
  width: number
  height: number
}
export type RectExt = Rect & { right: number; bottom: number }

export type PaddingVert = {
  top: number
  bottom: number
}
export type PaddingHorz = {
  left: number
  right: number
}
export type Padding = PaddingVert & PaddingHorz
export type Inset = Padding

export const newPadding = (
  x:
    | { top: number; bottom: number; left: number; right: number }
    | { vert: number; left: number; right: number }
    | { top: number; bottom: number; horz: number }
    | { vert: number; horz: number }
    | number,
): Padding => ({ ...newPaddingHorz(x), ...newPaddingVert(x) })

export const newPaddingHorz = (
  x: number | { horz: number } | { left: number; right: number },
) =>
  typeof x === 'number'
    ? { left: x, right: x }
    : 'horz' in x
    ? { left: x.horz, right: x.horz }
    : { left: x.left, right: x.right }

export const newPaddingVert = (
  x: number | { vert: number } | { top: number; bottom: number },
) =>
  typeof x === 'number'
    ? { top: x, bottom: x }
    : 'vert' in x
    ? { top: x.vert, bottom: x.vert }
    : { top: x.top, bottom: x.bottom }

export type InsetExt = Inset & { width: number; height: number }
export const insetExt = (spec: Inset | RectExt, parentSize: Size): InsetExt => {
  if ('width' in spec) {
    return insetExt(
      {
        top: spec.y,
        left: spec.x,
        right: parentSize.width - spec.right,
        bottom: parentSize.height - spec.bottom,
      },
      parentSize,
    )
  }
  const { top, left, right, bottom } = spec
  return {
    top,
    left,
    right,
    bottom,
    width: parentSize.width - left - right,
    height: parentSize.height - top - bottom,
  }
}

export const rectExt = (
  rect:
    | { x: number; y: number; width: number; height: number }
    | { x: number; y: number; width: number; bottom: number }
    | { x: number; y: number; right: number; height: number }
    | { x: number; y: number; right: number; bottom: number },
): RectExt => {
  const width = 'width' in rect ? rect.width : rect.right - rect.x
  const height = 'height' in rect ? rect.height : rect.bottom - rect.y
  const right = 'right' in rect ? rect.right : rect.width + rect.x
  const bottom = 'bottom' in rect ? rect.bottom : rect.height + rect.y
  const { x, y } = rect
  return { x, y, width, height, bottom, right }
}

rectExt.inset = (rect: RectExt, inset: Padding) =>
  rectExt({
    x: rect.x + inset.left,
    y: rect.y + inset.top,
    right: rect.right - inset.right,
    bottom: rect.bottom - inset.bottom,
  })

rectExt.translate = (rect: RectExt, translate: XY) =>
  rectExt({
    x: rect.x + translate.x,
    y: rect.y + translate.y,
    width: rect.width,
    height: rect.height,
  })

export const applyRegionToHTMLElement = (
  { x, y, width, height }: RectExt,
  element: HTMLElement,
) => {
  applyOriginToHTMLElement({ x, y }, element)
  applySizeToHTMLElement({ width, height }, element)
}

export const regionCSSStyle = (region: RectExt) => ({
  ...originCSSStyle(region),
  ...sizeCSSStyle(region),
})

export const applySizeToHTMLElement = (
  { width, height }: Size,
  element: HTMLElement,
) => {
  element.style.width = `${width}px`
  element.style.height = `${height}px`
}

export const sizeCSSStyle = ({ width, height }: Size) => ({
  width: `${width}px`,
  height: `${height}px`,
})

export const applyOriginToHTMLElement = (
  { x, y }: XY,
  element: HTMLElement,
) => {
  element.style.top = `${y}px`
  element.style.left = `${x}px`
}
export const originCSSStyle = ({ x, y }: XY) => ({
  left: `${x}px`,
  top: `${y}px`,
})
export const insetCSSStyle = ({ top, left, right, bottom }: Inset) => ({
  left: `${left}px`,
  top: `${top}px`,
  right: `${right}px`,
  bottom: `${bottom}px`,
})

export const applyHorzPaddingToHTMLElement = (
  { left, right }: { left: number; right: number },
  element: HTMLElement,
) => {
  element.style.paddingLeft = `${left}px`
  element.style.paddingRight = `${right}px`
}

export const applyPaddingToHTMLElement = (
  { left, right, top, bottom }: Padding,
  element: HTMLElement,
) => {
  applyHorzPaddingToHTMLElement({ left, right }, element)
  element.style.paddingTop = `${top}px`
  element.style.paddingBottom = `${bottom}px`
}

export const paddingCSS = ({ top, left, right, bottom }: Padding) =>
  `${top}px ${right}px ${bottom}px ${left}px`

export const paddingCSSStyleVert = (
  { top, bottom }: PaddingVert,
  { scale = 1 }: { scale?: number } = {},
) => ({
  paddingTop: `${top * scale}px`,
  paddingBottom: `${bottom * scale}px`,
})
export const paddingCSSStyleHorz = (
  { left, right }: PaddingHorz,
  { scale = 1 }: { scale?: number } = {},
) => ({
  paddingLeft: `${left * scale}px`,
  paddingRight: `${right * scale}px`,
})
export const paddingCSSStyle = (
  padding: Padding,
  { scale }: { scale?: number } = {},
) => ({
  ...paddingCSSStyleHorz(padding, { scale }),
  ...paddingCSSStyleVert(padding, { scale }),
})

export type Size = { width: number; height: number }
export type XY = { x: number; y: number }

export const clampPoint = ({ x, y }: XY, rect: RectExt) => ({
  x: _.clamp(x, rect.x, rect.right),
  y: _.clamp(y, rect.y, rect.bottom),
})
