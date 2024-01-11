import Head from 'next/head'
import React, { ReactNode, useState } from 'react'
import { Header } from './Header'

type Props = {
  title: string
  children:
    | ReactNode
    | ((x: { setDarkHeader: (x: boolean) => void }) => ReactNode)
  style?: React.CSSProperties
  className?: string
  isHeaderAPortal?: boolean
  onClick?: () => void
}
export const AppPage = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    (
      {
        title = '',
        children,
        className = '',
        style,
        onClick,
        isHeaderAPortal = false,
      }: Props,
      ref,
    ) => {
      const [darkHeader, setDarkHeader] = useState(false)
      return (
        <div
          className={`${className} page relative z-0`}
          style={style}
          ref={ref}
          onClick={onClick}
        >
          <Head>
            <title>{title}</title>
          </Head>
          {typeof children === 'function'
            ? children({ setDarkHeader })
            : children}
          <Header isDark={darkHeader} isPortal={isHeaderAPortal} />
        </div>
      )
    },
  ),
)
