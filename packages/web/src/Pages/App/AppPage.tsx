import Head from 'next/head'
import React, { ReactNode } from 'react'
import { Header } from './Header'

type Props = {
  title: string
  children: ReactNode
  style?: React.CSSProperties
  className?: string
  onClick?: () => void
}
export const AppPage = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    (
      { title = '', children, className = '', style, onClick }: Props,
      ref,
    ) => {
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
          {children}
          <Header />
        </div>
      )
    },
  ),
)
