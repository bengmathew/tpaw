import Head from 'next/head'
import Link from 'next/link'
import React, {ReactElement} from 'react'
import {Footer} from './Footer'

export const AppPageDark = React.memo(
  ({children, title}: {children: ReactElement; title: string}) => {
    return (
      <div
        className="font-font1 min-h-screen grid text-gray-200 bg-gray-800"
        style={{grid: 'auto 1fr auto/auto'}}
      >
        <Head>
          <title>{title}</title>
        </Head>
        <div className="w-full flex flex-col justify-center items-center my-2">
          <Link href={'/'}>
            <a className="flex justify-center items-end gap-x-2">
              <h2 className="font-semibold text-3xl sm:text-4xl ">TPAW</h2>
              <h2 className="font-semibold text-3xl sm:text-4xl ">Planner</h2>
            </a>
          </Link>
        </div>
        <div className="flex justify-center">
          <div className=" w-full max-w-[1000px] overflow-visible p-2 sm:p-4 bg-white text-gray-700 rounded-xl">
            {children}
          </div>
        </div>
        <Footer className="flex justify-center my-2  gap-x-4 sm:gap-x-4" />
      </div>
    )
  }
)
