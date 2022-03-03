import {GetStaticProps} from 'next'
import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {createContext} from '../../Utils/CreateContext'
import {useWindowSize} from '../../Utils/WithWindowSize'
import {AppPage} from '../App/AppPage'
import {useChartPanel} from './ChartPanel/ChartPanel'
import {ParamsInput} from './ParamsInput/ParamsInput'

type _FetchedInline = Awaited<ReturnType<typeof Contentful.fetchInline>>
type _IntroAndBody = {
  intro: _FetchedInline
  body: _FetchedInline
}
type _IntroAndBodyAndMenu = _IntroAndBody & {menu: _FetchedInline}
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
  chart: {
    spending: {
      total: _IntroAndBodyAndMenu
      regular: _IntroAndBodyAndMenu
      discretionary: _IntroAndBodyAndMenu
      essential: _IntroAndBodyAndMenu
    }
    portfolio: _IntroAndBodyAndMenu
    glidePath: _IntroAndBodyAndMenu
    withdrawalRate: _IntroAndBodyAndMenu
  }
}
const [PlanContentContext, usePlanContent] =
  createContext<PlanContent>('PlanContent')
export {usePlanContent}

export const Plan = React.memo((planContent: PlanContent) => {
  const {width, height} = useWindowSize()

  const aspectRatio = width / height
  const isPortrait = aspectRatio < 1.1
  const [setChartType, chartType, chartPanel] = useChartPanel({
    className: isPortrait
      ? 'h-[350px] sm:h-[425px] md:h-[500px] lg:h-[550px] w-full px-3 border-b-4 border-gray-600'
      : 'px-3',
    isPortrait,
  })
  return (
    <PlanContentContext.Provider value={planContent}>
      <AppPage
        className="h-screen grid"
        title="TPAW Planner"
        curr="plan"
        style={{
          grid: isPortrait
            ? '"chart" auto "params" 1fr /1fr'
            : '"params chart" 1fr /.6fr 1fr',
        }}
      >
        {chartPanel}
        <ParamsInput
          className=""
          bgClassName={isPortrait ? 'bg-pageBG' : 'bg-gray-100'}
          isPortrait={isPortrait}
          chartType={chartType}
          setChartType={setChartType}
        />
      </AppPage>
    </PlanContentContext.Provider>
  )
})

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
      chart: {
        spending: {
          total: {
            intro: await Contentful.fetchInline('6MH8oPq7ivMYJ4Ii8ZMtwg'),
            body: await Contentful.fetchInline('4tlCDgSlcKXfO8hfmZvBoF'),
            menu: await Contentful.fetchInline('21U0B92yz78PBR2YlX1CJP'),
          },
          regular: {
            intro: await Contentful.fetchInline('3e3y2gADSUszUxSqWvC2EV'),
            body: await Contentful.fetchInline('14KuXhGuaRok2dW0ppyArU'),
            menu: await Contentful.fetchInline('2tmRByrkc7dPNVOS63CUy3'),
          },
          discretionary: {
            intro: await Contentful.fetchInline('3pcwRLnI1BaBm0Hqv0Zyvg'),
            body: await Contentful.fetchInline('4QRDrEzc3uJHNI4SzRJ6r7'),
            menu: await Contentful.fetchInline('7F7JGrPs7ShQ2G4GyEkJ75'),
          },
          essential: {
            intro: await Contentful.fetchInline('3oWMTqbZARYjiZ2G9fuwTb'),
            body: await Contentful.fetchInline('4Vd9Ay5NnEodIMiBl83Vfs'),
            menu: await Contentful.fetchInline('7xqfxGtRc4pHZFJvR9zaY1'),
          },
        },
        portfolio: {
          intro: await Contentful.fetchInline('5KzxtC01WrdXnahFd98zet'),
          body: await Contentful.fetchInline('2iQeojVfV3Fw18WV4TRATR'),
          menu: await Contentful.fetchInline('7cRJH6TdeEPpcom84B8Dch'),
        },
        glidePath: {
          intro: await Contentful.fetchInline('247ji3WlKEyiMcthzbcaUX'),
          body: await Contentful.fetchInline('1r6sBwyo6ulnzLQwQOM05L'),
          menu: await Contentful.fetchInline('8MHwQEcXljZlEIcZrQRKd'),
        },
        withdrawalRate: {
          intro: await Contentful.fetchInline('7nDVZLSFxZcdHWmuSqzo6o'),
          body: await Contentful.fetchInline('79KDyYdPfxwl7BceHPECCe'),
          menu: await Contentful.fetchInline('42bT4OaF5u9GXukOHXHnWz'),
        },
      },
    },
  })
