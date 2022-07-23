export type Rect = {
  x: number
  y: number
  width: number
  height: number
}
export type RectExt = Rect & {right: number; bottom: number}

export type PaddingVert = {
  top: number
  bottom: number
}
export type PaddingHorz = {
  left: number
  right: number
}
export type Padding = PaddingVert & PaddingHorz

export const rectExt = (
  rect:
    | {x: number; y: number; width: number; height: number}
    | {x: number; y: number; width: number; bottom: number}
    | {x: number; y: number; right: number; height: number}
    | {x: number; y: number; right: number; bottom: number}
): RectExt => {
  const width = 'width' in rect ? rect.width : rect.right - rect.x
  const height = 'height' in rect ? rect.height : rect.bottom - rect.y
  const right = 'right' in rect ? rect.right : rect.width + rect.x
  const bottom = 'bottom' in rect ? rect.bottom : rect.height + rect.y
  const {x, y} = rect
  return {x, y, width, height, bottom, right}
}

export const applyRegionToHTMLElement = (
  {x, y, width, height}: RectExt,
  element: HTMLElement
) => {
  applyOriginToHTMLElement({x, y}, element)
  applySizeToElement({width, height}, element)
}

export const applySizeToElement = (
  {width, height}: Size,
  element: HTMLElement
) => {
  element.style.width = `${width}px`
  element.style.height = `${height}px`
}

export const sizeCSSStyle = ({width, height}: Size) => ({
  width: `${width}px`,
  height: `${height}px`,
})

export const applyOriginToHTMLElement = (
  {x, y}: Origin,
  element: HTMLElement
) => {
  element.style.top = `${y}px`
  element.style.left = `${x}px`
}
export const applyHorzPaddingToHTMLElement = (
  {left, right}: {left: number; right: number},
  element: HTMLElement
) => {
  element.style.paddingLeft = `${left}px`
  element.style.paddingRight = `${right}px`
}

export const applyPaddingToHTMLElement = (
  {left, right, top, bottom}: Padding,
  element: HTMLElement
) => {
  applyHorzPaddingToHTMLElement({left, right}, element)
  element.style.paddingTop = `${top}px`
  element.style.paddingBottom = `${bottom}px`
}

export const paddingCSS = ({top, left, right, bottom}: Padding) =>
  `${top}px ${right}px ${bottom}px ${left}px`

export const paddingCSSStyleVert = (
  {top, bottom}: PaddingVert,
  {scale = 1}: {scale?: number} = {}
) => ({
  paddingTop: `${top * scale}px`,
  paddingBottom: `${bottom * scale}px`,
})
export const paddingCSSStyleHorz = (
  {left, right}: PaddingHorz,
  {scale = 1}: {scale?: number} = {}
) => ({
  paddingLeft: `${left * scale}px`,
  paddingRight: `${right * scale}px`,
})
export const paddingCSSStyle = (
  padding: Padding,
  {scale}: {scale?: number} = {}
) => ({
  ...paddingCSSStyleHorz(padding, {scale}),
  ...paddingCSSStyleVert(padding, {scale}),
})

export type Size = {width: number; height: number}
export type Origin = {x: number; y: number}
