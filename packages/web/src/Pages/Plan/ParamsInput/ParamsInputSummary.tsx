import {faExclamation} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap, Power1} from 'gsap'
import _ from 'lodash'
import React, {useEffect, useRef} from 'react'
import {Transition} from 'react-transition-group'
import {TPAWParams, ValueForYearRange} from '../../../TPAWSimulator/TPAWParams'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {fGet, noCase} from '../../../Utils/Utils'
import {Footer} from '../../App/Footer'
import {useSimulation} from '../../App/WithSimulation'
import { ChartUtils } from '../ChartPanel/Chart/ChartUtils'
import {byYearScheduleYearRangeToStr} from './ByYearSchedule/ByYearSchedule'
import {paramsInputValidate} from './Helpers/ParamInputValidate'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {ParamsInputType} from './Helpers/ParamsInputType'
import {Reset} from './Reset'
import {Share} from './Share'

export const ParamsInputSummary = React.memo(
  ({
    isOpen,
    highlight,
    setState,
    allowSplit,
    bgClassName,
    showFooter,
    duration,
    displacement,
  }: {
    isOpen: boolean
    highlight: ParamsInputType | null
    setState: (state: ParamsInputType) => void
    allowSplit: boolean
    bgClassName: string
    showFooter: boolean
    duration: number
    displacement: number
  }) => {
    const {params} = useSimulation()
    const summaryRef = useRef<HTMLDivElement | null>(null)
    const isRetired = params.age.start === params.age.retirement
    return (
      <Transition
        in={isOpen}
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
            x: -displacement,
            duration,
          })
        }}
      >
        {tstate => (
          <div
            className={`${
              allowSplit
                ? 'px-2 sm:px-4 md:px-8 lg:px-[calc(50vw-496px)]'
                : 'px-4'
            } 
            grid
            absolute w-full h-full top-0  pb-4 overflow-scroll 
            ${tstate === 'exited' ? 'opacity-0' : ''}`} // This is needed if we start in exited state.
            ref={summaryRef}
            style={{grid: '1fr auto/1fr'}}
          >
            <div className="flex flex-col items-start">
              <div className="self-stretch flex justify-end sticky top-0 z-10 ">
                <div
                  className={`flex gap-x-4  py-2 bg-opacity-80 rounded-full ${bgClassName}`}
                >
                  <Reset />
                  <Share />
                </div>
              </div>
              <div className="flex flex-col gap-y-2 relative z-0">
                <_Button type="age" {...{setState, highlight}} />
                <div className="mt-2">
                  <h2 className="font-medium text-base">Savings and Income</h2>
                  <div className="flex flex-col gap-y-2 mt-4 ml-8">
                    <_Button
                      type="current-portfolio-value"
                      {...{setState, highlight}}
                    />
                    {(!isRetired || params.savings.length > 0) && (
                      <_Button
                        type="future-savings"
                        {...{setState, highlight}}
                        warn={!paramsInputValidate(params, 'futureSavings')}
                      />
                    )}
                    <_Button
                      type="income-during-retirement"
                      {...{setState, highlight}}
                      warn={!paramsInputValidate(params, 'retirementIncome')}
                    />
                  </div>
                </div>
                <div className="mt-2">
                  <h2 className="font-medium text-base">Spending</h2>
                  <div className="flex flex-col gap-y-2 mt-4 ml-8">
                    <_Button
                      type="extra-spending"
                      {...{setState, highlight}}
                      warn={!paramsInputValidate(params, 'extraSpending')}
                    />
                    <_Button
                      type="spending-ceiling-and-floor"
                      {...{setState, highlight}}
                    />
                    <_Button type="legacy" {...{setState, highlight}} />
                  </div>
                </div>

                <div className="mt-2">
                  <h2 className="font-medium text-base">
                    Expected Returns and Inflation
                  </h2>
                  <div className="flex flex-col gap-y-2 mt-4 ml-8">
                    <_Button
                      type="expected-returns"
                      {...{setState, highlight}}
                    />
                    <_Button type="inflation" {...{setState, highlight}} />
                  </div>
                </div>

                <_Button
                  type="risk-and-time-preference"
                  {...{setState, highlight}}
                />
              </div>
            </div>

            {showFooter && (
              <Footer className="flex justify-center  gap-x-4 mt-8 lighten-2" />
            )}
          </div>
        )}
      </Transition>
    )
  }
)

const _Button = React.memo(
  ({
    type,
    setState,
    warn = false,
    highlight,
    className = '',
  }: {
    type: ParamsInputType
    highlight: ParamsInputType | null
    setState: (state: ParamsInputType) => void
    warn?: boolean
    className?: string
  }) => {
    const {params} = useSimulation()
    const ref = useRef<HTMLButtonElement | null>(null)
    useEffect(() => {
      if (highlight !== type) return
      gsap.fromTo(
        ref.current,
        {backgroundColor: `${ChartUtils.color.gray[300]}FF`},
        {backgroundColor: `${ChartUtils.color.gray[300]}00`, duration: 1, ease:Power1.easeIn}
      )
    }, [highlight, type])

    return (
      <button
        className={`${className} text-left p-1  rounded-md`}
        onClick={() => setState(type)}
        ref={ref}
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

        {_.flatten([_text(type, params)]).map((text, i) => (
          <h2 key={i} className="text-sm lighten-2">
            {text}
          </h2>
        ))}
      </button>
    )
  }
)

export const _text = (type: ParamsInputType, params: TPAWParams) => {
  switch (type) {
    case 'age': {
      const retired = params.age.start === params.age.retirement
      return retired
        ? `Retired, Current: ${params.age.start}, Max: ${params.age.end}`
        : `Current: ${params.age.start}, Retirement: ${params.age.retirement}, Max: ${params.age.end}`
    }
    case 'risk-and-time-preference': {
      const format = formatPercentage(1)
      return `Stock Allocation: ${format(
        params.targetAllocation.regularPortfolio.stocks
      )}, Spending Tilt: ${format(params.scheduledWithdrawalGrowthRate)}`
    }
    case 'current-portfolio-value': {
      return `${formatCurrency(params.savingsAtStartOfStartYear)}`
    }
    case 'future-savings':
      return _entriesSummary(params.savings)
    case 'income-during-retirement':
      return _entriesSummary(params.retirementIncome)
    case 'extra-spending':
      return _entriesSummary([
        ...params.withdrawals.fundedByBonds,
        ...params.withdrawals.fundedByRiskPortfolio,
      ])
    case 'spending-ceiling-and-floor': {
      return params.spendingCeiling === null && params.spendingFloor === null
        ? 'None'
        : params.spendingCeiling === params.spendingFloor
        ? `Fixed Spending: ${formatCurrency(fGet(params.spendingCeiling))}`
        : _.compact([
            params.spendingCeiling === null
              ? undefined
              : `Ceiling: ${formatCurrency(params.spendingCeiling)}`,
            ,
            params.spendingFloor === null
              ? undefined
              : `Floor: ${formatCurrency(params.spendingFloor)}`,
          ]).join(', ')
    }
    case 'legacy': {
      return `${formatCurrency(params.legacy.total)}`
    }
    case 'expected-returns': {
      const format = formatPercentage(1)
      return `Stocks: ${format(
        params.returns.expected.stocks
      )}, Bonds: ${format(params.returns.expected.bonds)}`
    }
    case 'inflation': {
      const format = formatPercentage(1)
      return `${format(params.inflation)}`
    }
    default:
      noCase(type)
  }
}

const _entriesSummary = (entries: ValueForYearRange[]) => {
  const n = entries.length
  return n === 0
    ? 'None'
    : n <= 4
    ? entries.map(_byYearEntryToStr)
    : [
        ...entries.slice(0, 3).map(_byYearEntryToStr),
        `and ${n - 3} more entries.`,
      ]
}

const _byYearEntryToStr = ({label, value, yearRange}: ValueForYearRange) =>
  `${label ?? '<no label>'} - ${formatCurrency(
    value
  )} ${byYearScheduleYearRangeToStr(yearRange)}`
