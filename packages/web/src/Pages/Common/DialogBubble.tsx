import React, { ReactNode, useEffect, useLayoutEffect, useState } from 'react'

export const DialogBubble = React.memo(
  ({
    arrowOffset,
    arrowDirection,
    x,
    y,
    zIndex,
    children,
  }: {
    arrowOffset: number
    arrowDirection: 'up' | 'down'
    x: number
    y: number
    zIndex: number
    children: ReactNode
  }) => {
    const [element, setElement] = useState<HTMLDivElement | null>(null)
    const [height, setHeight] = useState(0)
    useLayoutEffect(() => {
      if (!element) return
      const rs = new ResizeObserver((entries) => {
        // Not using entires because not available on iOS 14.
        setHeight(element.getBoundingClientRect().height)
      })
      rs.observe(element, { box: 'border-box' })
    }, [element])

    return (
      <div
        ref={setElement}
        className="absolute page text-lg bg-orange-400  text-gray-700 px-5 py-2 rounded-lg"
        style={{
          left: `${x - arrowOffset}px`,
          top: arrowDirection === 'up' ? `${y + 10}px` : `${y - 10 - height}px`,
          zIndex: `${zIndex}`,
        }}
      >
        <div
          className={`absolute w-[20px] h-[20px] rounded-sm  bg-orange-400
          ${
            arrowDirection === 'up'
              ? 'origin-top-left top-[-10px] rotate-45'
              : 'origin-bottom-left bottom-[-10px] -rotate-45'
          }`}
          style={{ left: `${arrowOffset}px` }}
        />
        <div className="relative z-10">{children}</div>
      </div>
    )
  },
)
