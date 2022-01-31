import {GetStaticProps} from 'next'
import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {createContext} from '../../Utils/CreateContext'
import {useWindowSize} from '../../Utils/WithWindowSize'
import {AppPage} from '../App/AppPage'
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
      <AppPage
        className="h-screen grid"
        title="TPAW Planner"
        curr="plan"
        style={{
          grid: isPortrait
            ? '"chart" auto "params" 1fr /1fr'
            : '"params chart" 1fr /1fr 2fr',
        }}
      >
        {isPortrait ? (
          <ChartPanel
            className="h-[350px] sm:h-[400px] md:h-[470px] lg:h-[525px] w-full px-3 border-b-4 border-gray-700"
            isPortrait
          />
        ) : (
          <ChartPanel className="px-3" isPortrait={false} />
        )}
        {isPortrait ? (
          <ParamsInput
            className=""
            bgClassName="bg-pageBG"
            allowSplit
          />
        ) : (
          <ParamsInput
            className=""
            bgClassName="bg-gray-100"
            
            allowSplit={false}
          />
        )}
      </AppPage>
    </PlanContentContext.Provider>
  )
})
