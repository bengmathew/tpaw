import Head from 'next/head'
import Link from 'next/link'
import React from 'react'
import {Footer} from '../App/Footer'

export const Index = React.memo(() => {
  return (
    <div
      className="font-font1 min-h-screen grid text-gray-800"
      style={{grid: '1fr auto/auto'}}
    >
      <Head>
        <title>TPAW Planner</title>
      </Head>
      <div className="flex flex-col justify-center items-center px-2">
        <div className="max-w-[700px]">
          <h1 className="font-bold text-3xl sm:text-4xl ">TPAW Planner</h1>
          <p className="mt-6 text-lg">
            TPAW Planner is a retirement planner that uses the Total Portfolio
            Allocation and Withdrawal (TPAW) strategy to calculate asset
            allocation and withdrawal.{' '}
            <Link href="/about">
              <a href="" className="underline font-semibold">
                Learn more.
              </a>
            </Link>
          </p>
          <div className=" mt-10">
            <Link href="/plan">
              <a className="text-xl px-4 py-2 btn-dark">Make Your Plan</a>
            </Link>
          </div>
        </div>
      </div>
      <Footer className="flex justify-center my-2  gap-x-4 sm:gap-x-4" />
    </div>
  )
})
