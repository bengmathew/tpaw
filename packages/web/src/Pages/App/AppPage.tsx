import Head from 'next/head'
import Link from 'next/link'
import React, {ReactElement} from 'react'

export const AppPage = React.memo(
  ({children, title}: {children: ReactElement; title: string}) => {
    return (
      <div
        className="font-font1 min-h-screen grid"
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
          <div className=" w-full max-w-[1200px] overflow-hidden px-2 sm:px-10 md:px-20 mt-8 mb-4 ">
            {children}
          </div>
        </div>
        <div className="flex justify-center my-4  gap-x-4 sm:gap-x-4">
          <Link href="/about"><a className="">About</a></Link>
          <Link href="/license"><a className="">License</a></Link>
          <Link href="/disclaimer"><a className="">Disclaimer</a></Link>
          <Link href="/privacy"><a className="">Privacy</a></Link>
        </div>
      </div>
    )
  }
)
