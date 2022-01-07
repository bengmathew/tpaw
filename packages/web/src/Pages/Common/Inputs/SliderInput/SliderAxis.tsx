import React from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { lineFromPoints } from '../../../../Utils/SVG/LineFromPoints'
import { SVGSizing } from '../../../../Utils/SVG/UseSVGSizing'

export const SliderAxis = React.memo(
  ({
    sizing: {scale, xDomain},
    activeRange
  }: {
    sizing: SVGSizing
    activeRange:SimpleRange
  }) => {
    return (
      <g>
        <path
          className="stroke-current text-gray-200"
          strokeWidth={1}
          d={lineFromPoints([
            [0, 0],
            [scale.x(activeRange.start), 0],
          ])}
        />
        <path
          className="stroke-current text-gray-400"
          strokeWidth={1}
          d={lineFromPoints([
            [scale.x(activeRange.start), 0],
            [scale.x(activeRange.end), 0],
          ])}
        />
        <path
          className="stroke-current text-gray-200"
          strokeWidth={1}
          d={lineFromPoints([
            [scale.x(activeRange.end), 0],
            [scale.x(xDomain.end), 0],
          ])}
        />
      </g>
    )
  }
)
