import clsx from 'clsx'
import React, { CSSProperties, ReactNode } from 'react'

export const PrintSection = React.memo(
  ({
    children,
    className,
    style,
  }: {
    children: ReactNode
    className?: string
    style?: CSSProperties
  }) => {
    return (
      <div
        className={clsx(
          className,
          'w-[21cm]',
          'min-h-[27.9cm] print:min-h-[100vh]',
          'border border-gray-200 print:border-0',
          'rounded-lg print:rounded-none',
          'shadow-lg print:shadow-none',
          // When printing browser (chrome at least) adds a margin to the top and bottom of the
          // page by default.
          'px-[1.5cm] py-[1.5cm] print:py-[.5cm]',
          'my-10 print:my-0',
          'bg-white',
          'text-[12px]',

          // 'w-[21cm]',
          // 'min-h-[100vh]',
          // 'border-x border-gray-200 print:border-x-0',
          // 'px-[1.5cm] py-[1.5cm]',
          // 'bg-white',
          // 'text-[14px]',
        )}
        style={{ ...style, pageBreakAfter: 'always' }}
      >
        {children}
      </div>
    )
  },
)