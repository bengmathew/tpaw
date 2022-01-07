import {faArrowAltLeft, faExclamation} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap} from 'gsap'
import Link from 'next/link'
import React, {Dispatch, useRef, useState} from 'react'
import {Transition} from 'react-transition-group'
import {noCase} from '../../../Utils/Utils'
import {Footer} from '../../App/Footer'
import {useSimulation} from '../../App/WithSimulation'
import {paramsInputValidate} from './Helpers/ParamInputValidate'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {paramsInputSummary} from './Helpers/ParamsInputSummary'
import {ParamsInputTypes} from './Helpers/ParamsInputTypes'
import {ParamsInputAge} from './ParamsInputAge'
import {ParamsInputCurrentPortfolioValue} from './ParamsInputCurrentPortfolioValue'
import {ParamsInputExtraSpending} from './ParamsInputExtraSpending'
import {ParamsInputFutureSavings} from './ParamsInputFutureSavings'
import {ParamsInputLegacy} from './ParamsInputLegacy'
import {ParamsInputRetirementIncome} from './ParamsInputRetirementIncome'
import {ParamsInputReturnsAndInflation} from './ParamsInputReturnsAndInflation'
import {ParamsInputSpendingCeiling} from './ParamsInputSpendingCeiling'
import {Reset} from './Reset'
import {Share} from './Share'

type _State = 'summary' | ParamsInputTypes

const duration = 0.3
const displacement = 30

export const ParamsInput = React.memo(
  ({className = '', showFooter}: {className?: string; showFooter: boolean}) => {
    const [state, setState] = useState<_State>('summary')
    const {params} = useSimulation()

    const summaryRef = useRef<HTMLDivElement | null>(null)

    const isRetired = params.age.start === params.age.retirement

    return (
      <div
        className={`${className} relative overflow-hidden h-full bg-inherit`}
      >
        <Transition
          in={state === 'summary'}
          timeout={duration * 1000}
          onEntering={() => {
            gsap.fromTo(
              summaryRef.current,
              {opacity: 0, x: displacement},
              {opacity: 1, x: 0, duration}
            )
          }}
          onExiting={() => {
            gsap.to(summaryRef.current, {
              opacity: 0,
              x: displacement,
              duration,
            })
          }}
        >
          <div
            className={`absolute h-full w-full  top-0  pb-4 overflow-scroll bg-inherit`}
            ref={summaryRef}
          >
            <div className="flex justify-end sticky top-0 z-10  bg-inherit ">
              <div className="flex gap-x-4  py-2 bg-opacity-80 rounded-full  bg-inherit ">
                <Reset />
                <Share />
              </div>
            </div>
            <div className="flex flex-col gap-y-4 relative z-0">
              <_Button type="age" setState={setState} />
              <div className="mt-2">
                <h2 className="font-bold text-base">Savings and Income</h2>
                <div className="flex flex-col gap-y-4 mt-4 ml-4">
                  <_Button type="currentPortfolioValue" setState={setState} />
                  {(!isRetired || params.savings.length > 0) && (
                    <_Button
                      type="futureSavings"
                      setState={setState}
                      warn={!paramsInputValidate(params, 'futureSavings')}
                    />
                  )}
                  <_Button
                    type="retirementIncome"
                    setState={setState}
                    warn={!paramsInputValidate(params, 'retirementIncome')}
                  />
                </div>
              </div>
              <div className="mt-2">
                <h2 className="font-bold text-base">Spending</h2>
                <div className="flex flex-col gap-y-4 mt-4 ml-4">
                  <_Button
                    type="extraSpending"
                    setState={setState}
                    warn={!paramsInputValidate(params, 'extraSpending')}
                  />
                  <_Button type="spendingCeiling" setState={setState} />
                  <_Button type="legacy" setState={setState} />
                </div>
              </div>
              <_Button type="expectedReturnsAndInflation" setState={setState} />
            </div>
            {showFooter && (
              <Footer className="flex justify-center  gap-x-4 mt-8 lighten-2" />
            )}
          </div>
        </Transition>
        <_Detail type="age" {...{state, setState}} />
        <_Detail type="currentPortfolioValue" {...{state, setState}} />
        <_Detail type="futureSavings" {...{state, setState}} />
        <_Detail type="retirementIncome" {...{state, setState}} />
        <_Detail type="extraSpending" {...{state, setState}} />
        <_Detail type="spendingCeiling" {...{state, setState}} />
        <_Detail type="legacy" {...{state, setState}} />
        <_Detail type="expectedReturnsAndInflation" {...{state, setState}} />
      </div>
    )
  }
)
const _Button = React.memo(
  ({
    type,
    setState,
    warn = false,
    className = '',
  }: {
    type: ParamsInputTypes
    setState: Dispatch<_State>
    warn?: boolean
    className?: string
  }) => {
    const {params} = useSimulation()

    return (
      <button
        className={`${className} text-left`}
        onClick={() => setState(type)}
      >
        <div className=" flex items-center">
          <h2 className="font-medium text-base sm:text-base ">
            {paramsInputLabel(type)}
          </h2>
          {warn && (
            <h2 className="h-[20px] w-[20px] flex items-center justify-center ml-2 text-[11px] rounded-full bg-errorBlockBG text-errorBlockFG">
              <FontAwesomeIcon icon={faExclamation} />
            </h2>
          )}
        </div>
        <h2 className="text-sm lighten-2">
          {paramsInputSummary(type, params)}
        </h2>
      </button>
    )
  }
)

const _Detail = React.memo(
  ({
    type,
    state,
    setState,
  }: {
    type: ParamsInputTypes
    state: _State
    setState: Dispatch<_State>
  }) => {
    const detailRef = useRef<HTMLDivElement | null>(null)

    const Body = React.memo(() => {
      switch (type) {
        case 'age':
          return <ParamsInputAge />
        case 'currentPortfolioValue':
          return <ParamsInputCurrentPortfolioValue />
        case 'futureSavings':
          return <ParamsInputFutureSavings onBack={() => setState('summary')} />
        case 'retirementIncome':
          return <ParamsInputRetirementIncome />
        case 'extraSpending':
          return <ParamsInputExtraSpending />
        case 'spendingCeiling':
          return <ParamsInputSpendingCeiling />
        case 'legacy':
          return <ParamsInputLegacy />
        case 'expectedReturnsAndInflation':
          return <ParamsInputReturnsAndInflation />
        default:
          noCase(type)
      }
    })

    return (
      <Transition
        in={state === type}
        timeout={duration * 1000}
        mountOnEnter
        unmountOnExit
        onEntering={() => {
          gsap.fromTo(
            detailRef.current,
            {opacity: 0, x: -displacement},
            {opacity: 1, x: 0, duration}
          )
        }}
        onExiting={() => {
          gsap.to(detailRef.current, {opacity: 0, x: -displacement, duration})
        }}
      >
        <div
          className={`absolute top-0 w-full h-full grid bg-inherit`}
          style={{grid: 'auto 1fr/auto'}}
          ref={detailRef}
        >
          <div className=" bg-inherit">
            <button
              className="py-3 flex items-center gap-x-2 "
              onClick={() => setState('summary')}
            >
              <FontAwesomeIcon className="text-lg" icon={faArrowAltLeft} />{' '}
              <span className="text-xl">Back</span>
            </button>
          </div>
          <div className="overflow-scroll pb-4">
            <h2 className="font-bold text-lg  mb-2">
              {paramsInputLabel(type)}
            </h2>
            <div className="max-w-[500px]">
              <Body />
            </div>
          </div>
        </div>
      </Transition>
    )
  }
)
