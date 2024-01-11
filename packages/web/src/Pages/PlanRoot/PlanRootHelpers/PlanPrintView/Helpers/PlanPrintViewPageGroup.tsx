import { noCase } from '@tpaw/common'
import clsx from 'clsx'
import React, { CSSProperties, ReactNode } from 'react'
import { useSystemInfo } from '../../../../App/WithSystemInfo'
import { PlanPrintViewArgs } from '../PlanPrintViewArgs'

export const PlanPrintViewPageGroup = React.memo(
  ({
    children,
    className,
    style,
    settings,
  }: {
    children: ReactNode
    className?: string
    style?: CSSProperties
    settings: PlanPrintViewArgs['settings']
  }) => {
    const { pageSize } = settings
    return (
      <div
        className={clsx(
          className,
          'bg-white print:bg-none',
          // Horizontal margins have to be set here and not using css@page
          // margin property because chart sizing does not work correctly
          // otherwiser.
          'px-[1in]  ',
          // Vertical margins have to set using css @page margin property
          // because otherwise content that spans page breaks will not have any
          // vertical margins at the page break. Since they are set in css they
          // have to be mimicked here, and removed on printing when @page will
          // take over.
          'py-[1in] print:py-0',
          pageSize === 'A4'
            ? 'min-h-[29.7cm] '
            : pageSize === 'Letter'
              ? 'min-h-[27.94cm] print:min-h-[100vh]'
              : noCase(pageSize),
          'border border-gray-300 print:border-0',
          'rounded-lg print:rounded-none',
          'shadow-lg print:shadow-none',
          ' break-after-page last:break-after-auto',
        )}
        style={{ ...style }}
      >
        {children}
      </div>
    )
  },
)
