import clsx from 'clsx'
import React from 'react'

export const PlanPrintSVGBackground = React.memo(
  ({ className, fill }: { className?: string; fill: string }) => {
    return (
      <svg
        className={(clsx(className), 'absolute h-full w-full')}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
      >
        <rect x={0} y={0} width={1} height={1} fill={fill}></rect>
      </svg>
    )
  },
)
