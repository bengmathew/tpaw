export type Rect = {
  x: number
  y: number
  width: number
  height: number
}
export type RectExt = Rect & {right: number; bottom: number}

export type Padding = {
  left: number
  right: number
  top: number
  bottom: number
}

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

export const applyRectSizingToHTMLElement = (
  {x, y, width, height}: RectExt,
  element: HTMLElement
) => {
  applyPositionToHTMLElement({x, y}, element)
  element.style.width = `${width}px`
  element.style.height = `${height}px`
}
export const applyPositionToHTMLElement = (
  {x, y}: {x: number; y: number},
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

export type Size = {width: number; height: number}
