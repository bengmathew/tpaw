import {gsap} from 'gsap'
import _ from 'lodash'
import {GetStaticProps} from 'next'
import Head from 'next/head'
import {useRouter} from 'next/router'
import React, {useEffect, useMemo, useRef, useState} from 'react'
import {Contentful} from '../../Utils/Contentful'
import {createContext} from '../../Utils/CreateContext'
import {rectExt} from '../../Utils/Geometry'
import {linearFnFomPoints} from '../../Utils/LinearFn'
import {useURLParam} from '../../Utils/UseURLParam'
import {fGet, noCase} from '../../Utils/Utils'
import {useWindowSize} from '../../Utils/WithWindowSize'
import {AppPage} from '../App/AppPage'
import {Footer} from '../App/Footer'
import {headerHeight} from '../App/Header'
import {useSimulation} from '../App/WithSimulation'
import {ChartPanel, ChartPanelStateful} from './ChartPanel/ChartPanel'
import {chartPanelLabel} from './ChartPanel/ChartPanelLabel'
import {useChartPanelState} from './ChartPanel/UseChartPanelState'
import {GuidePanel, GuidePanelStateful} from './GuidePanel'
import {paramsInputLabel} from './ParamsInput/Helpers/ParamsInputLabel'
import {
  isParamsInputType,
  ParamsInputType,
} from './ParamsInput/Helpers/ParamsInputType'
import {ParamsInput, ParamsInputStateful} from './ParamsInput/ParamsInput'
import {PlanHeading, PlanHeadingStateful} from './PlanHeading'

const duration = 300 / 1000

type _State = 'summary' | ParamsInputType
export const Plan = React.memo((planContent: PlanContent) => {
  const windowSize = useWindowSize()

  const headingRef = useRef<PlanHeadingStateful | null>(null)
  const paramsRef = useRef<ParamsInputStateful | null>(null)
  const guideRef = useRef<GuidePanelStateful | null>(null)
  const chartRef = useRef<ChartPanelStateful | null>(null)
  const aspectRatio = windowSize.width / windowSize.height
  const layout =
    aspectRatio > 1.1
      ? 'laptop'
      : windowSize.width <= 700
      ? 'mobile'
      : 'desktop'

  const simulation = useSimulation()
  const stateInStr = useURLParam('input') ?? 'summary'
  const stateIn = isParamsInputType(stateInStr) ? stateInStr : 'summary'
  const [state, setStateLocal] = useState<_State>(stateIn)

  const router = useRouter()
  const [highlight, setHighlight] = useState<ParamsInputType | null>(null)
  const setState = (newState: _State) => {
    setStateLocal(newState)
    const url = new URL(window.location.href)
    if (newState === 'summary') {
      if (state !== 'summary') setHighlight(state)
      url.searchParams.delete('input')
    } else {
      setHighlight(null)
      url.searchParams.set('input', newState)
    }
    void router.push(url)
  }

  const [prevState, setPrevState] = useState<ParamsInputType>('age')
  useEffect(() => {
    if (state !== 'summary') setPrevState(state)
  }, [state])

  const paramType = state === 'summary' ? prevState : state

  const transitionRef = useRef({transition: state === 'summary' ? 0 : 1})
  const _sizing = useMemo(() => {
    switch (layout) {
      // ------------------------------------------
      // -------------- LAPTOP --------------------
      // ------------------------------------------
      case 'laptop': {
        const pad = 30
        const topFn = (transition: number) =>
          linearFnFomPoints(0, 50, 1, 70)(transition)

        const headingMarginBottom = 20

        // Guide
        const guide = (transition: number) => {
          const y = topFn(1)
          const height = windowSize.height - y
          const width = windowSize.width * 0.28
          const x = linearFnFomPoints(0, -width + pad * 0.75, 1, 0)(transition)
          return {
            position: rectExt({width, x, height, y}),
            padding: {left: pad, right: pad * 0.75, top: 0, bottom: pad},
            headingMarginBottom,
          }
        }

        // Input
        const input = (transition: number) => {
          const horzCardPadding = linearFnFomPoints(0, 10, 1, 20)(transition)
          const vertCardPadding = linearFnFomPoints(0, 10, 1, 20)(transition)
          const totalTop = topFn(transition)
          const y = linearFnFomPoints(0, 0, 1, totalTop)(transition)
          const top = totalTop - y
          const height = windowSize.height - y
          const x = guide(transition).position.right
          const width = Math.max(
            windowSize.width * linearFnFomPoints(0, 0.37, 1, 0.3)(transition),
            350
          )
          return {
            position: rectExt({width, x, height, y}),
            padding: {
              left: pad * 0.25,
              right: pad * 0.75,
              top,
              bottom: pad,
            },
            cardPadding: {
              left: horzCardPadding,
              right: horzCardPadding,
              top: vertCardPadding,
              bottom: vertCardPadding,
            },
            headingMarginBottom,
          }
        }

        // Heading
        const heading = (transition: number) => {
          return {
            position: rectExt({
              width: input(1).position.right - pad,
              height: 60,
              x: guide(transition).position.x + pad,
              y: 0,
            }),
          }
        }

        // Chart
        const chart = (transition: number) => {
          const padTop = 15
          const positionFn = (transition: number) => {
            const y = topFn(transition) - padTop
            const x = input(transition).position.right + pad * .25
            return {
              y,
              x,
              width: windowSize.width - x - pad,
              height: windowSize.height - 2 * y,
            }
          }

          const position = positionFn(transition)
          const positionAt0 = positionFn(0)
          const positionAt1 = positionFn(1)

          const heightAt1 = Math.min(
            positionAt1.height,
            Math.max(400, positionAt1.width * 0.85)
          )
          position.height = linearFnFomPoints(
            0,
            positionAt0.height,
            1,
            heightAt1
          )(transition)

          return {
            position: rectExt(position),
            padding: {
              left: 20,
              right: 20,
              top: 10,
              bottom: linearFnFomPoints(0, 10, 1, 0)(transition),
            },
            // 18px is text-lg 24px is text-2xl.
            menuButtonScale: linearFnFomPoints(0, 1, 1, 18 / 24)(transition),
            cardPadding: {left: 10, right: 10, top: 10, bottom: 10},
            headingMarginBottom: 10,
          }
        }

        return {input, guide, chart, heading}
      }

      // ------------------------------------------
      // -------------- DESKTOP --------------------
      // ------------------------------------------
      case 'desktop': {
        const pad = 20

        const navHeadingH = 60
        const navHeadingMarginBottom = 20
        const chart = (transition: number) => {
          const inset = 0
          let height =
            windowSize.width < 800
              ? linearFnFomPoints(400, 300, 800, 500)(windowSize.width)
              : 500

          const padBottom = (transition: number) =>
            linearFnFomPoints(0, pad, 1, pad / 2)(transition)
          height -= linearFnFomPoints(
            0,
            0,
            1,
            (navHeadingH + navHeadingMarginBottom) / 2
          )(transition)
          return {
            position: rectExt({
              width: windowSize.width - inset * 2,
              height: height,
              x: inset,
              y: inset,
            }),
            padding: {
              left: pad * 2,
              right: pad * 2,
              top: pad,
              bottom: padBottom(transition),
            },
            cardPadding: {
              left: 10,
              right: 10,
              top: 10,
              bottom: 10,
            },
            headingMarginBottom,
            menuButtonScale: 1,
          }
        }

        const heading = (transition: number) => {
          return {
            position: rectExt({
              width: windowSize.width - pad * 2,
              height: navHeadingH,
              x: pad * 2 + linearFnFomPoints(0, 0, 1, 0)(transition),
              y: chart(1).position.bottom,
            }),
          }
        }

        const input = (transition: number) => {
          const horzBodyPadding = linearFnFomPoints(0, 10, 1, 20)(transition)
          const y = linearFnFomPoints(
            0,
            chart(0).position.bottom,
            1,
            heading(1).position.bottom
          )(transition)
          return {
            position: rectExt({
              width: linearFnFomPoints(
                0,
                550,
                1,
                windowSize.width * 0.5
              )(transition),
              height: windowSize.height - y,
              x: 0,
              y,
            }),
            padding: {
              left: pad * 2,
              right: pad * 1.75,
              top: navHeadingMarginBottom,
              bottom: linearFnFomPoints(0, 0, 1, pad)(transition),
            },
            cardPadding: {
              left: horzBodyPadding,
              right: horzBodyPadding,
              top: 10,
              bottom: 10,
            },
            headingMarginBottom,
          }
        }

        const headingMarginBottom = 10

        const guide = (transition: number) => {
          const inputPositionAt1 = input(1).position
          return {
            position: rectExt({
              width: windowSize.width - inputPositionAt1.right,
              height: inputPositionAt1.height,
              x: inputPositionAt1.right,
              y: input(transition).position.y,
            }),
            padding: {
              left: pad * 0.25,
              right: pad * 2,
              top: navHeadingMarginBottom,
              bottom: pad,
            },
            cardPadding: {left: 0, right: 0, top: 0, bottom: 0},
            headingMarginBottom,
          }
        }

        return {input, guide, chart, heading}
      }
      // ------------------------------------------
      // -------------- MOBILE --------------------
      // ------------------------------------------
      case 'mobile': {
        const pad = 10

        const navHeadingH = 40
        const navHeadingMarginBottom = 10
        const chart = (transition: number) => {
          let height = linearFnFomPoints(375, 330, 415, 355)(windowSize.width)

          const padBottom = (transition: number) =>
            linearFnFomPoints(0, pad, 1, 0)(transition)
          height -= linearFnFomPoints(
            0,
            0,
            1,
            (navHeadingH + navHeadingMarginBottom) / 2
          )(transition)
          return {
            position: rectExt({
              width: windowSize.width,
              height: height,
              x: 0,
              y: 0,
            }),
            padding: {
              left: pad,
              right: pad,
              top: headerHeight + 5,
              bottom: padBottom(transition),
            },
            cardPadding: {
              left: 15,
              right: 15,
              top: 10,
              bottom: 10,
            },
            headingMarginBottom,
            menuButtonScale: 1,
          }
        }

        const heading = (transition: number) => {
          return {
            position: rectExt({
              width: windowSize.width - pad * 2,
              height: navHeadingH,
              x: pad * 2,
              y: chart(1).position.bottom + 10,
            }),
          }
        }

        const input = (transition: number) => {
          const horzBodyPadding = linearFnFomPoints(0, 10, 1, 20)(transition)
          const y = linearFnFomPoints(
            0,
            chart(0).position.bottom,
            1,
            heading(1).position.bottom
          )(transition)
          return {
            position: rectExt({
              width: windowSize.width,
              height: windowSize.height - y,
              x: 0,
              y,
            }),
            padding: {
              left: linearFnFomPoints(0, pad, 1, pad * 2)(transition),
              right: pad,
              top: navHeadingMarginBottom,
              bottom: 0,
            },
            cardPadding: {
              left: horzBodyPadding,
              right: horzBodyPadding,
              top: 10,
              bottom: 10,
            },
            headingMarginBottom: 10,
          }
        }

        const headingMarginBottom = 10

        const guide = (transition: number) => {
          const pad = 10
          const height = 50
          const width = 100
          return {
            position: rectExt({
              width,
              height,
              x: windowSize.width - width - pad,
              y:
                windowSize.height -
                height -
                pad +
                linearFnFomPoints(0, height + pad, 1, 0)(transition),
            }),
            padding: {left: pad, right: pad, top: pad, bottom: pad},
            cardPadding: {left: 0, right: 0, top: 0, bottom: 0},
            headingMarginBottom: 0,
          }
        }
        return {input, guide, chart, heading}
      }
      default:
        noCase(layout)
    }
  }, [layout, windowSize])

  const chartPanelState = useChartPanelState()
  // This indirection for dev ergonomics because in dev update is very slow
  // when saving state directly in URL.
  useEffect(() => {
    setState(stateIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateIn])
  const chartLabel = chartPanelLabel(
    simulation.params,
    chartPanelState.state.type,
    'full'
  )

  useEffect(() => {
    const tween = gsap.to(transitionRef.current, {
      transition: state === 'summary' ? 0 : 1,
      duration,
      onUpdate: function () {
        const {transition} = this.targets()[0] as {transition: number}
        fGet(headingRef.current).setTransition(transition)
        fGet(paramsRef.current).setTransition(transition)
        fGet(guideRef.current).setTransition(transition)
        fGet(chartRef.current).setTransition(transition)
      },
    })
    return () => {
      tween.kill()
    }
  }, [state])

  return (
    <PlanContentContext.Provider value={planContent}>
      <Head>
        <title>
          Plan
          {chartPanelState.state.type === 'spending-total'
            ? ''
            : ` - View:${_.compact([
                ...chartLabel.label,
                chartLabel.subLabel,
              ]).join(' - ')}`}
          {state === 'summary' ? '' : ` - Input: ${paramsInputLabel(state)}`} -
          TPAW Planner
        </title>
      </Head>
      <AppPage
        className="h-screen  bg-planBG overflow-hidden"
        title="TPAW Planner"
        curr="plan"
      >
        <PlanHeading
          sizing={_sizing.heading}
          transitionRef={transitionRef}
          type={paramType}
          ref={headingRef}
          onDone={() => setState('summary')}
        />
        <ChartPanel
          layout={layout}
          state={chartPanelState}
          sizing={_sizing.chart}
          ref={chartRef}
          transitionRef={transitionRef}
        />
        <ParamsInput
          layout={layout}
          sizing={_sizing.input}
          transitionRef={transitionRef}
          paramInputType={paramType}
          state={state}
          setState={setState}
          chartType={chartPanelState.state.type}
          setChartType={chartPanelState.handleChangeType}
          ref={paramsRef}
        />
        <GuidePanel
          key={state}
          layout={layout}
          sizing={_sizing.guide}
          transitionRef={transitionRef}
          type={paramType}
          ref={guideRef}
        />
        {layout === 'laptop' && (
          <div className="absolute right-0 -bottom-0 mr-[35px]">
            <Footer />
          </div>
        )}
      </AppPage>
    </PlanContentContext.Provider>
  )
})

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

export const planGetStaticProps: GetStaticProps<
  PlanContent
> = async context => ({
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
