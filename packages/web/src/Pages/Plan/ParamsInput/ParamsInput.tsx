import {Document} from '@contentful/rich-text-types'
import {faTimes} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap} from 'gsap'
import _ from 'lodash'
import Head from 'next/head'
import {useRouter} from 'next/router'
import React, {useEffect, useRef, useState} from 'react'
import {Transition} from 'react-transition-group'
import {Contentful} from '../../../Utils/Contentful'
import {useURLParam} from '../../../Utils/UseURLParam'
import {noCase} from '../../../Utils/Utils'
import {useWindowSize} from '../../../Utils/WithWindowSize'
import {Footer} from '../../App/Footer'
import {useSimulation} from '../../App/WithSimulation'
import {chartPanelLabel} from '../ChartPanel/ChartPanelLabel'
import {ChartPanelType} from '../ChartPanel/ChartPanelType'
import {usePlanContent} from '../Plan'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {
  isParamsInputType,
  ParamsInputType,
  paramsInputTypes,
} from './Helpers/ParamsInputType'
import {ParamsInputAge} from './ParamsInputAge'
import {ParamsInputCurrentPortfolioValue} from './ParamsInputCurrentPortfolioValue'
import {ParamsInputExpectedReturns} from './ParamsInputExpectedReturns'
import {ParamsInputExtraSpending} from './ParamsInputExtraSpending'
import {ParamsInputFutureSavings} from './ParamsInputFutureSavings'
import {ParamsInputIncomeDuringRetirement} from './ParamsInputIncomeDuringRetirement'
import {ParamsInputInflation} from './ParamsInputInflation'
import {ParamsInputLegacy} from './ParamsInputLegacy'
import {ParamsInputRiskAndTimePreference} from './ParamsInputRiskAndTimePreference'
import {ParamsInputSpendingCeilingAndFloor} from './ParamsInputSpendingCeilingAndFloor'
import {ParamsInputSummary} from './ParamsInputSummary'

type _State = 'summary' | ParamsInputType

const duration = 0.3
const displacement = 30

export const ParamsInput = React.memo(
  ({
    className = '',
    bgClassName,
    isPortrait,
    chartType,
    setChartType,
  }: {
    className?: string
    bgClassName: string
    isPortrait: boolean
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  }) => {
    const simulation = useSimulation()
    const stateInStr = useURLParam('input') ?? 'summary'
    const stateIn = isParamsInputType(stateInStr) ? stateInStr : 'summary'
    // const state: _State = isParamsInputType(stateIn) ? stateIn : 'summary'
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

    // This indirection for dev ergonomics because in dev update is very slow
    // when saving state directly in URL.
    useEffect(() => {
      setState(stateIn)
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stateIn])
    const chartLabel = chartPanelLabel(
      simulation.params,
      chartType,
      'full'
    )

    return (
      <div
        className={`${className} ${bgClassName} relative overflow-hidden h-full `}
        style={{gridArea: 'params'}}
      >
        <Head>
          <title>
            Plan
            {chartType === 'spending-total'
              ? ''
              : ` - View:${_.compact([...chartLabel.label, chartLabel.subLabel]).join(' - ')}`}
            {state === 'summary' ? '' : ` - Input: ${paramsInputLabel(state)}`}{' '}
            - TPAW Planner
          </title>
        </Head>
        <ParamsInputSummary
          {...{
            isOpen: state === 'summary',
            highlight,
            setState,
            duration,
            displacement,
            allowSplit: isPortrait,
          }}
          allowSplit={isPortrait}
        />
        {paramsInputTypes.map((type, i) => (
          <_Detail
            key={i}
            allowSplit={isPortrait}
            type={type}
            state={state}
            setState={setState}
            bgClassName={bgClassName}
            chartType={chartType}
            setChartType={setChartType}
          />
        ))}
      </div>
    )
  }
)

const _Detail = React.memo(
  ({
    type,
    state,
    setState,
    bgClassName,
    allowSplit,
    chartType,
    setChartType,
  }: {
    type: ParamsInputType
    state: _State
    setState: (state: ParamsInputType | 'summary') => void
    bgClassName: string
    allowSplit: boolean
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  }) => {
    const detailRef = useRef<HTMLDivElement | null>(null)
    const content = useContent(type)
    const windowSize = useWindowSize()
    const onDone = () => setState('summary')

    return (
      <Transition
        in={state === type}
        timeout={duration * 1000}
        mountOnEnter
        unmountOnExit
        onEntering={() => {
          gsap.fromTo(
            detailRef.current,
            {opacity: 0, x: displacement},
            {opacity: 1, x: 0, duration}
          )
        }}
        onExiting={() => {
          gsap.to(detailRef.current, {opacity: 0, x: -displacement, duration})
        }}
      >
        {allowSplit ? (
          windowSize.width > 900 ? (
            <div
              className={`absolute top-0 h-full w-full grid`}
              style={{grid: '1fr/1fr 1fr'}}
              ref={detailRef}
            >
              <div className={`overflow-scroll pr-8 plan-pl pb-10`}>
                <_Heading
                  className={`sticky top-0 z-10 mb-6 ${bgClassName} bg-opacity-90`}
                  {...{type, setState}}
                />
                <_Body {...{type, onDone, chartType, setChartType}} />
              </div>
              <div
                className={`grid pt-4 overflow-scroll bg-gray-200  `}
                style={{grid: '1fr auto / 1fr'}}
              >

                <div className="pl-8 plan-pr pt-4  h-full opacity-90 mb-20">
                <h2 className="font-bold uppercase  text-sm -mt-2 mb-4 text-gray-500">Instructions</h2>
                  <_RichText className="">{content.body.fields.body}</_RichText>
                </div>
                <Footer />
              </div>
            </div>
          ) : (
            <div
              className={`absolute top-0 h-full w-full overflow-scroll grid`}
              style={{grid: 'auto auto 1fr / 1fr'}}
              ref={detailRef}
            >
              <div
                className={`sticky top-0 plan-pl plan-pr z-10  mb-6 ${bgClassName} bg-opacity-90`}
              >
                <_Heading className="" {...{type, setState}} />
              </div>

              <div className="plan-pl plan-pr pb-8">
                <_Body {...{type, onDone, chartType, setChartType}} />
              </div>
              <div
                className={`bg-gray-200  pt-4 opacity-90 plan-pl plan-pr grid`}
                style={{grid: 'auto 1fr auto/1fr'}}
              >
                <h2 className="font-bold uppercase  text-sm -mt-2 mb-4 text-gray-500">Instructions</h2>
                <_RichText className="pb-20">
                  {content.body.fields.body}
                </_RichText>
                <Footer />
              </div>
            </div>
          )
        ) : (
          <div
            className={`absolute top-0 h-full w-full grid`}
            style={{grid: 'auto minmax(45vh, 1fr)/1fr'}}
            ref={detailRef}
          >
            <div className={`overflow-scroll px-10 pt-2 pb-8`}>
              <_Heading
                className={`sticky top-0 z-10  mb-6 ${bgClassName} bg-opacity-90`}
                {...{type, setState}}
              />
              <_Body {...{type, onDone, chartType, setChartType}} />
            </div>
            <div className="bg-gray-200 border-t-2 border-gray-600  pt-8 opacity-90 overflow-scroll px-10 grid" 
                style={{grid: 'auto 1fr auto/1fr'}}>
                <h2 className="font-bold uppercase  text-sm -mt-2 mb-4 text-gray-500">Instructions</h2>
              <_RichText className={`pb-16`}>
                {content.body.fields.body}
              </_RichText>
              <Footer />
            </div>
          </div>
        )}
      </Transition>
    )
  }
)

const _Body = React.memo(
  ({
    type,
    onDone,
    chartType,
    setChartType,
  }: {
    type: ParamsInputType
    onDone: () => void
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  }) => {
    switch (type) {
      case 'age':
        return <ParamsInputAge />
      case 'risk-and-time-preference':
        return <ParamsInputRiskAndTimePreference />
      case 'current-portfolio-value':
        return <ParamsInputCurrentPortfolioValue />
      case 'future-savings':
        return <ParamsInputFutureSavings onBack={onDone} />
      case 'income-during-retirement':
        return <ParamsInputIncomeDuringRetirement />
      case 'extra-spending':
        return (
          <ParamsInputExtraSpending
            chartType={chartType}
            setChartType={setChartType}
          />
        )
      case 'spending-ceiling-and-floor':
        return <ParamsInputSpendingCeilingAndFloor />
      case 'legacy':
        return <ParamsInputLegacy />
      case 'expected-returns':
        return <ParamsInputExpectedReturns />
      case 'inflation':
        return <ParamsInputInflation />
      default:
        noCase(type)
    }
  }
)

const _RichText = React.memo(
  ({className = '', children}: {className?: string; children: Document}) => {
    return (
      <div className={`${className}`}>
        <Contentful.RichText
          body={children}
          ul="list-disc ml-5"
          ol="list-decimal ml-5"
          p="p-base mb-3"
          h1="font-bold text-lg mb-3"
          h2="font-bold text-lg mt-6 mb-3"
          a="underline"
          aExternalLink="text-[12px] ml-1"
        />
      </div>
    )
  }
)

const _Heading = React.memo(
  ({
    className = '',
    type,
    setState,
  }: {
    className?: string
    type: ParamsInputType
    setState: (state: ParamsInputType | 'summary') => void
  }) => {
    return (
      <div
        className={`relative grid items-center ${className} `}
        style={{grid: 'auto/40px 1fr 40px'}}
      >
        <button
          className="text-2xl px-6 -ml-6 py-2"
          onClick={() => setState('summary')}
        >
          <FontAwesomeIcon icon={faTimes} />
        </button>
        <h2 className="font-bold text-lg sm:text-xl text-center ">
          {paramsInputLabel(type)}
        </h2>
      </div>
    )
  }
)

function useContent(type: ParamsInputType) {
  const content = usePlanContent()
  switch (type) {
    case 'age':
      return content.age
    case 'risk-and-time-preference':
      return content.riskAndTimePreference
    case 'current-portfolio-value':
      return content.currentPortfolioValue
    case 'future-savings':
      return content.futureSavings
    case 'income-during-retirement':
      return content.incomeDuringRetirement
    case 'extra-spending':
      return content.extraSpending
    case 'spending-ceiling-and-floor':
      return content.spendingCeilingAndFloor
    case 'legacy':
      return content.legacy
    case 'expected-returns':
      return content.expectedReturns
    case 'inflation':
      return content.inflation
    default:
      noCase(type)
  }
}
