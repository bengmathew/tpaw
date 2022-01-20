import {GetStaticProps} from 'next'
import Head from 'next/head'
import Link from 'next/link'
import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {createContext} from '../../Utils/CreateContext'
import {useWindowSize} from '../../Utils/WithWindowSize'
import {Footer} from '../App/Footer'
import {ChartPanel} from './ChartPanel/ChartPanel'
import {ParamsInput} from './ParamsInput/ParamsInput'

type _FetchedInline = Awaited<ReturnType<typeof Contentful.fetchInline>>
type _IntroAndBody = {
  intro: _FetchedInline
  body: _FetchedInline
}
export type PlanContent = {
  riskAndTimePreference: _IntroAndBody
  age: {
    introRetired: _FetchedInline
    introNotRetired: _FetchedInline
    body: _FetchedInline
  }
  currentPortfolioValue: _IntroAndBody
  incomeDuringRetirement: _IntroAndBody
  futureSavings: _IntroAndBody
  extraSpending: _IntroAndBody
  spendingCeilingAndFloor: _IntroAndBody
  legacy: {
    introAmount: _FetchedInline
    introAssets: _FetchedInline
    body: _FetchedInline
  }
  expectedReturns: _IntroAndBody
  inflation: _IntroAndBody
}
const [PlanContentContext, usePlanContent] =
  createContext<PlanContent>('PlanContent')
export {usePlanContent}

export const planGetStaticProps: GetStaticProps<PlanContent> =
  async context => ({
    props: {
      age: {
        introRetired: await Contentful.fetchInline('1dZPTbtQfLz3cyDrMGrAQB'),
        introNotRetired: await Contentful.fetchInline('43EyTxBVHWOPcA6rgBpsnG'),
        body: await Contentful.fetchInline('5EtkcdtSIg0rS8AETEsgnm'),
      },
      currentPortfolioValue: {
        intro: await Contentful.fetchInline('3iLyyrQAhHnzuc4IdftWT3'),
        body: await Contentful.fetchInline('5RE7wTwvtTAsF1sWpKrFW2'),
      },
      riskAndTimePreference: {
        intro: await Contentful.fetchInline('4UHaDSQWXjNW75yTAwK1IX'),
        body: await Contentful.fetchInline('3ofgPmJFLgtJpjl26E7jpB'),
      },
      incomeDuringRetirement: {
        intro: await Contentful.fetchInline('3OqUTPDVRGzgQcVkJV7Lew'),
        body: await Contentful.fetchInline('1MHvhL8ImdOL9FxE5qxK6F'),
      },
      futureSavings: {
        intro: await Contentful.fetchInline('2rPr5mMTcScftXhletDeb4'),
        body: await Contentful.fetchInline('5aJN2Z4tZ7zQ6Tw69VelRt'),
      },
      extraSpending: {
        intro: await Contentful.fetchInline('01kv7sKzniBagrcIwX86tJ'),
        body: await Contentful.fetchInline('5zDvtk4dDOonIkoIyOeQH8'),
      },
      spendingCeilingAndFloor: {
        intro: await Contentful.fetchInline('19Llaw2GVZhEfBTfGzE7Ns'),
        body: await Contentful.fetchInline('6hEbQkY7ctTpMpGV6fBBu2'),
      },
      legacy: {
        introAmount: await Contentful.fetchInline('aSdQuriQu9ztfs812MRJj'),
        introAssets: await Contentful.fetchInline('5glA8ryQcNh7SHP9ZlkZ2y'),
        body: await Contentful.fetchInline('5nCHpNy6ReAEtBQTvDTwBf'),
      },
      expectedReturns: {
        intro: await Contentful.fetchInline('2NxIclWQoxuk0TMVH0GjhR'),
        body: await Contentful.fetchInline('2GxHf6q4kfRrz6AnFLniFh'),
      },
      inflation: {
        intro: await Contentful.fetchInline('76BgIpwX9yZetMGungnfwC'),
        body: await Contentful.fetchInline('6LqbR3PBA1uDe9xU2V1hk9'),
      },
    },
  })

export const Plan = React.memo((planContent: PlanContent) => {
  const {width, height} = useWindowSize()

  const aspectRatio = width / height
  const isPortrait = aspectRatio < 1.1
  return (
    <PlanContentContext.Provider value={planContent}>
      <div className="font-font1 h-screen text-gray-800 ">
        <Head>
          <title>TPAW Planner</title>
        </Head>
        <div
          className=" h-full grid "
          style={{
            grid: isPortrait
              ? '"chart" auto "params" 1fr /1fr'
              : '"params chart" 1fr /1fr 2fr',
          }}
        >
          {isPortrait ? (
            <div
              className=" flex flex-col items-center border-b-4 border-gray-700 plan-pl plan-pr"
              style={{gridArea: 'chart'}}
            >
              <Link href="/">
                <a className="font-bold  text-xl md:text-3xl mt-1 py-1 sm:py-2 ">
                  TPAW Planner
                </a>
              </Link>
              <ChartPanel className=" w-full " />
            </div>
          ) : (
            <div
              className="grid min-h-full overflow-scroll px-4"
              style={{grid: 'auto 1fr auto/1fr', gridArea: 'chart'}}
            >
              <div className="flex justify-center mb-8">
                <Link href="/">
                  <a className="text-3xl font-bold my-2">TPAW Planner</a>
                </Link>
              </div>
              <div className="flex flex-col justify-center">
                <ChartPanel className=" " />
              </div>
              <Footer className="flex justify-center  gap-x-4 mt-8 lighten-2 mb-2" />
            </div>
          )}
          {isPortrait ? (
            <ParamsInput
              className=""
              bgClassName="bg-pageBG"
              showFooter
              allowSplit
            />
          ) : (
            <ParamsInput
              // className="border-r-4 border-gray-700"
              className=""
              bgClassName="bg-gray-100"
              showFooter={false}
              allowSplit={false}
            />
          )}
        </div>
      </div>
    </PlanContentContext.Provider>
  )
})
