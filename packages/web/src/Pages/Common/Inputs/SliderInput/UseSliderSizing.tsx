import { linearFnFomPoints } from '@tpaw/common'
import _ from 'lodash'
import { useMemo } from 'react'
import { PaddingHorz, rectExt } from '../../../../Utils/Geometry'

export type SliderSizing = ReturnType<typeof useSliderSizing>
export const useSliderSizing = (
  width: number,
  height: number,
  maxOverflowHorz: PaddingHorz,
  numPoints: number, // This is inclusive
) =>
  // Note. SVG coordinates has y axis flipped from a typical graph. Positive y
  // goes down, just as in HTML.
  // https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Positions
  useMemo(() => {
    const paddingHorz = height * 0.25
    const plotSize = {
      width: width - paddingHorz * 2,
      top: height / 2, // top area is 0 to -top
      bottom: height / 2, // bottom area is 0 to bottom
    }
    const viewBox = rectExt({
      x: -maxOverflowHorz.left - paddingHorz,
      y: -height / 2,
      width: width + maxOverflowHorz.left + maxOverflowHorz.right,
      height,
    })
    const viewBoxStr = `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`
    const svgProps = {
      width: viewBox.width,
      height: viewBox.height,
      viewBox: viewBoxStr,
    }

    const unclamped = linearFnFomPoints(0, 0, plotSize.width, numPoints - 1)
    const pixelXToDataX = (x: number) => _.clamp(unclamped(x), 0, numPoints - 1)
    pixelXToDataX.inverse = unclamped.inverse
    return { plotSize, viewBox, svgProps, pixelXToDataX, numPoints }
  }, [width, maxOverflowHorz, height, numPoints])
