import { scaleLinear } from '@visx/scale'
import { useMemo } from 'react'

export type SVGSizing = ReturnType<typeof useSVGSizing>
export const useSVGSizing = ({
  width,
  height,
  padding,
  xDomain,
  yDomain,
}: {
  width: number
  height: number
  padding: {left: number; right: number; top: number; bottom: number}
  xDomain: {start: number; end: number}
  yDomain: {start: number; end: number}
}) => {
  const viewBox = useMemo(
    () => ({
      x: -padding.left,
      y: -(height - padding.bottom),
      width,
      height,
    }),
    [height, padding.bottom, padding.left, width]
  )
  const viewBoxStr = `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`
  const target = useMemo(
    () => ({
      x: 0,
      y: -(height - padding.top - padding.bottom),
      width: width - padding.left - padding.right,
      height: height - padding.top - padding.bottom,
    }),
    [height, padding.bottom, padding.left, padding.right, padding.top, width]
  )
  const scale = useMemo(
    () => ({
      x: scaleLinear({
        domain: [xDomain.start, xDomain.end],
        range: [0, target.width],
      }),

      y: scaleLinear({
        domain: [yDomain.start, yDomain.end],
        range: [0, -target.height],
      }),
    }),
    [
      target.height,
      target.width,
      xDomain.end,
      xDomain.start,
      yDomain.end,
      yDomain.start,
    ]
  )

  return {viewBox, viewBoxStr, target, scale, padding, xDomain, yDomain}
}
