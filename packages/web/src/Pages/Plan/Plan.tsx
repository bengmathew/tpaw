import {gsap} from 'gsap'
import _ from 'lodash'
import Head from 'next/head'
import {useRouter} from 'next/router'
import React, {useEffect, useMemo, useRef, useState} from 'react'
import {createContext} from '../../Utils/CreateContext'
import {useURLParam} from '../../Utils/UseURLParam'
import {fGet} from '../../Utils/Utils'
import {useWindowSize} from '../App/WithWindowSize'
import {AppPage} from '../App/AppPage'
import {Footer} from '../App/Footer'
import {useSimulation} from '../App/WithSimulation'
import {ChartPanel, ChartPanelStateful} from './ChartPanel/ChartPanel'
import {chartPanelLabel} from './ChartPanel/ChartPanelLabel'
import {useChartPanelTypeState} from './ChartPanel/UseChartPanelTypeState'
import {GuidePanel, GuidePanelStateful} from './GuidePanel'
import {paramsInputLabel} from './ParamsInput/Helpers/ParamsInputLabel'
import {
  isParamsInputType,
  ParamsInputType,
} from './ParamsInput/Helpers/ParamsInputType'
import {ParamsInput, ParamsInputStateful} from './ParamsInput/ParamsInput'
import {
  ParamsInputSummary,
  ParamsInputSummaryStateful,
} from './ParamsInputSummary/ParamsInputSummary'
import {PlanContent} from './PlanGetStaticProps'
import {PlanHeading, PlanHeadingStateful} from './PlanHeading'
import {planSizing} from './PlanSizing/PlanSizing'

const duration = 300 / 1000

const [PlanContentContext, usePlanContent] =
  createContext<PlanContent>('PlanContent')
export {usePlanContent}

type _State = 'summary' | ParamsInputType
export const Plan = React.memo((planContent: PlanContent) => {


  const windowSizeIn = useWindowSize()
  // Hack to handle soft keyboard on Android.
  const windowSize = useMemo(() => {
    const {width, height} = windowSizeIn
    return {
      width,
      height: width < 600 ? Math.max(height, 700) : height,
    }
  }, [windowSizeIn])

  const headingRef = useRef<PlanHeadingStateful | null>(null)
  const paramsRef = useRef<ParamsInputStateful | null>(null)
  const inputSummaryRef = useRef<ParamsInputSummaryStateful | null>(null)
  const guideRef = useRef<GuidePanelStateful | null>(null)
  const chartRef = useRef<ChartPanelStateful | null>(null)
  const aspectRatio = windowSize.width / windowSize.height
  const layout =
    aspectRatio > 1.2
      ? 'laptop'
      : windowSize.width <= 700
      ? 'mobile'
      : 'desktop'

  const simulation = useSimulation()
  const stateInStr = useURLParam('input') ?? 'summary'
  const stateIn = isParamsInputType(stateInStr) ? stateInStr : 'summary'
  const [state, setStateLocal] = useState<_State>(stateIn)

  const router = useRouter()
  const setState = (newState: _State) => {
    setStateLocal(newState)
    const url = new URL(window.location.href)
    if (newState === 'summary') {
      url.searchParams.delete('input')
    } else {
      url.searchParams.set('input', newState)
    }
    void router.push(url)
  }

  const [prevState, setPrevState] =
    useState<ParamsInputType>('age-and-retirement')
  useEffect(() => {
    if (state !== 'summary') setPrevState(state)
  }, [state])

  const paramType = state === 'summary' ? prevState : state

  const transitionStart: 0 | 1 = state === 'summary' ? 0 : 1
  const transitionRef = useRef({
    transition: transitionStart,
    target: transitionStart,
  })
  const _sizing = useMemo(
    () => planSizing(layout, windowSize),
    [layout, windowSize]
  )

  const [chartPanelType, setChartPanelType] = useChartPanelTypeState()

  // This indirection for dev ergonomics because in dev update is very slow
  // when saving state directly in URL.
  useEffect(() => {
    setState(stateIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateIn])
  const chartLabel = chartPanelLabel(simulation.params, chartPanelType, 'full')

  useEffect(() => {
    const target = state === 'summary' ? 0 : 1
    transitionRef.current.target = target
    const tween = gsap.to(transitionRef.current, {
      transition: target,
      duration,
      onUpdate: function () {
        const {transition} = this.targets()[0] as {transition: number}
        fGet(headingRef.current).setTransition(transition)
        fGet(paramsRef.current).setTransition(transition)
        fGet(inputSummaryRef.current).setTransition(transition)
        fGet(guideRef.current).setTransition(transition)
        fGet(chartRef.current).setTransition(transition, target)
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
          {chartPanelType === 'spending-total'
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
          // state={chartPanelState}
          type={chartPanelType}
          setType={setChartPanelType}
          sizing={_sizing.chart}
          ref={chartRef}
          transitionRef={transitionRef}
        />
        <ParamsInputSummary
          layout={layout}
          state={state}
          setState={setState}
          sizing={_sizing.inputSummary}
          transitionRef={transitionRef}
          ref={inputSummaryRef}
        />
        <ParamsInput
          layout={layout}
          sizing={_sizing.input}
          transitionRef={transitionRef}
          paramInputType={paramType}
          setState={setState}
          chartType={chartPanelType}
          setChartType={setChartPanelType}
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
          <div className="absolute right-0 -bottom-0 mr-[45px] z-20">
            <Footer />
          </div>
        )}
      </AppPage>
    </PlanContentContext.Provider>
  )
})
