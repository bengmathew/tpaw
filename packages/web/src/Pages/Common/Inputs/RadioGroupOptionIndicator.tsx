import { noCase } from '@tpaw/common'
import clsx from 'clsx'
import React from 'react'

export const RadioGroupOptionIndicator = React.memo(
  ({
    className,
    size,
    selected,
  }: {
    className?: string
    size: 'base'
    selected: boolean
  }) => {
    const pixelSize = size === 'base' ? 16 : noCase(size)
    const border = size === 'base' ? 1 : noCase(size)
    const innerSize = pixelSize - border * 2
    const dotSize = size === 'base' ? 8 : noCase(size)
    return (
      <div
        className={clsx(className, 'inline-block relative rounded-full')}
        style={{
          width: `${pixelSize}px`,
          height: `${pixelSize}px`,
          borderWidth: `${border}px`,
          borderColor: 'currentcolor',
        }}
      >
        <div
          className={clsx('absolute rounded-full')}
          style={{
            transitionProperty: 'transform',
            transitionDuration: '300ms',
            transform: selected ? 'scale(1)' : 'scale(0)',
            width: `${dotSize}px`,
            height: `${dotSize}px`,
            left: `${(innerSize - dotSize) / 2}px`,
            top: `${(innerSize - dotSize) / 2}px`,
            backgroundColor: 'currentcolor',
          }}
        />
      </div>
    )
  },
)
