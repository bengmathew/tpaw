import Head from 'next/head'
import React, { ReactNode } from 'react'
import {Header} from './Header'

export const AppPage = React.memo(
  ({
    title = '',
    curr,
    children,
    className = '',
    style,
  }: {
    title: string
    children: ReactNode
    curr: 'plan' | 'learn' | 'other'
    style?: React.CSSProperties
    className?: string
  }) => {
    return (
      <div className={`${className} page`} style={style}>
        <Head>
          <title>{title}</title>
        </Head>
        {children}
        <Header curr={curr} />
      </div>
    )
  }
)
