import Head from 'next/head'
import Link from 'next/link'
import React from 'react'
import { useWindowSize } from '../../Utils/WithWindowSize'
import { ChartPanel } from './ChartPanel/ChartPanel'
import { ParamsInput } from './ParamsInput/ParamsInput'

export const Plan = React.memo(() => {
  const {width, height} = useWindowSize()
  const aspectRatio = width / height
  return (
    <div className="font-font1 h-screen text-gray-800 ">
      <Head>
        <title>TPAW Planner</title>
      </Head>
      {aspectRatio < 1.1 ? (
        <div className=" h-full  grid " style={{grid: ' auto   1fr /auto'}}>
          <div className=" flex flex-col items-center bg-white border-b-4 border-gray-700">
            {/* <div className="bg-white w-full flex justify-center py-2"> */}
              <Link href="/">
                <a className="font-bold  text-xl mt-1 px-4 ">TPAW Planner</a>
              </Link>
            {/* </div> */}
            <ChartPanel className=" w-full max-w-[1200px] px-2 sm:px-4 md:px-8 lg:px-20 " />
          </div>
          <div className="flex flex-col items-center bg-gray-100">
            <div className="w-full max-w-[1200px] px-2 sm:px-4 md:px-8 lg:px-20 h-full">
              <ParamsInput className="" showFooter />
            </div>
          </div>
        </div>
      ) : (
        <div
          className=" h-full w-full  grid  items-start"
          style={{grid: ' 1fr/1fr 2fr'}}
        >
          <div className="h-full px-4 bg-gray-100">
            <ParamsInput className="rounded-2xl " showFooter={false} />
          </div>

          <div className="h-full overflow-scroll  flex flex-col justify-center px-4">
            <div className="flex justify-center mb-8">
              <Link href="/">
                <a className="text-3xl font-bold ">
                  TPAW Planner
                </a>
              </Link>
            </div>
            <div className="flex flex-col justify-center">
              <ChartPanel className=" " />
            </div>
          </div>
        </div>
      )}
    </div>
  )
})
