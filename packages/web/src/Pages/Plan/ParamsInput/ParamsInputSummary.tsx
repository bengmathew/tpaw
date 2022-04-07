import {faExclamation} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap, Power1} from 'gsap'
import React, {useEffect, useRef} from 'react'
import {Transition} from 'react-transition-group'
import {
  Person,
  ValueForYearRange,
  Year,
} from '../../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {trimAndNullify} from '../../../Utils/TrimAndNullify'
import {fGet, noCase} from '../../../Utils/Utils'
import {Footer} from '../../App/Footer'
import {useSimulation} from '../../App/WithSimulation'
import {ChartUtils} from '../../Common/Chart/ChartUtils/ChartUtils'
import {ValueForYearRangeDisplay} from '../../Common/ValueForYearRangeDisplay'
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
    duration,
    displacement,
  }: {
    isOpen: boolean
    highlight: ParamsInputType | null
    setState: (state: ParamsInputType) => void
    allowSplit: boolean
    duration: number
    displacement: number
  }) => {
    const {params} = useSimulation()
    const {asYFN, withdrawalStartYear} = extendTPAWParams(params)
    const summaryRef = useRef<HTMLDivElement | null>(null)
    const isRetired = asYFN(withdrawalStartYear) <= 0

    return (
      <Transition
        in={isOpen}
        timeout={duration * 1000}
        onEntering={() => {
          gsap.fromTo(
            summaryRef.current,
            {opacity: 0, x: -displacement},
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
            className={`text-pageFGLight ${
              allowSplit ? 'plan-pr plan-pl' : 'px-8'
            } 
            grid
            absolute w-full h-full top-0  overflow-scroll 
            ${tstate === 'exited' ? 'opacity-0' : ''}`} // This is needed if we start in exited state.
            ref={summaryRef}
            style={{grid: '1fr auto/1fr'}}
          >
            <div className="flex flex-col items-start mb-16">
              <div className="self-stretch flex justify-end  ">
                <div className={`flex gap-x-4  py-2`}>
                  <Reset />
                  <Share />
                </div>
              </div>
              <div className="flex flex-col gap-y-2 relative z-0">
                <_Button type="age" {...{setState, highlight}} />
                <div className="mt-2">
                  <h2 className="font-semibold">Savings and Income</h2>
                  <div className="flex flex-col gap-y-2 mt-4 ml-8">
                    <_Button
                      type="current-portfolio-value"
                      {...{setState, highlight}}
                    />
                    {!isRetired && (
                      <_Button
                        type="future-savings"
                        {...{setState, highlight}}
                        warn={!paramsInputValidate(params, 'future-savings')}
                      />
                    )}
                    <_Button
                      type="income-during-retirement"
                      {...{setState, highlight}}
                      warn={
                        !paramsInputValidate(params, 'income-during-retirement')
                      }
                    />
                  </div>
                </div>
                <div className="mt-2">
                  <h2 className="font-semibold">Spending</h2>
                  <div className="flex flex-col gap-y-2 mt-4 ml-8">
                    <_Button
                      type="extra-spending"
                      {...{setState, highlight}}
                      warn={!paramsInputValidate(params, 'extra-spending')}
                    />
                    <_Button
                      type="spending-ceiling-and-floor"
                      {...{setState, highlight}}
                    />
                    <_Button type="legacy" {...{setState, highlight}} />
                  </div>
                </div>

                <div className="mt-2">
                  <h2 className="font-semibold">
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

            <Footer />
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
        {
          backgroundColor: `${ChartUtils.color.gray[300]}00`,
          duration: 1,
          ease: Power1.easeIn,
        }
      )
    }, [highlight, type])

    return (
      <button
        className={`${className} text-left p-1  rounded-md`}
        onClick={() => setState(type)}
        ref={ref}
      >
        <div className=" flex items-center">
          <h2 className="font-semibold  ">{paramsInputLabel(type)}</h2>
          {warn && (
            <h2 className="h-[18px] w-[18px] flex items-center justify-center ml-2 text-[13px] font-bold rounded-full bg-errorBlockBG text-errorBlockFG">
              <FontAwesomeIcon icon={faExclamation} />
            </h2>
          )}
        </div>
        <div className="grid gap-y-1 text-sm text-gray-500">
          <_SectionSummary type={type} />
        </div>
      </button>
    )
  }
)

export const _SectionSummary = React.memo(
  ({className = '', type}: {className?: string; type: ParamsInputType}) => {
    const {params, paramsExt} = useSimulation()
    const {validYearRange, pickPerson, yourOrYourPartners} = paramsExt

    switch (type) {
      case 'age': {
        const forPerson = ({ages}: Person) =>
          ages.type === 'retired'
            ? `Retired, Current: ${ages.current}, Max: ${ages.max}`
            : `Current: ${ages.current}, Retirement: ${ages.retirement}, Max: ${ages.max}`
        if (params.people.withPartner) {
          const withdrawalPerson = pickPerson(params.people.withdrawalStart)
          return (
            <>
              <h2 className="">
                <span className=" ">You</span> –{' '}
                {forPerson(params.people.person1)}
              </h2>
              <h2 className="">
                <span className=" ">Your Partner</span> –{' '}
                {forPerson(params.people.person2)}
              </h2>
              <h2 className="">
                Withdrawals start{' '}
                {withdrawalPerson.ages.type === 'retired'
                  ? 'now.'
                  : `at ${yourOrYourPartners(
                      params.people.withdrawalStart
                    )} retirement.`}
              </h2>
            </>
          )
        } else {
          return <h2>{forPerson(params.people.person1)}</h2>
        }
      }
      case 'risk-and-time-preference': {
        const format = formatPercentage(1)
        return (
          <h2>
            Stock Allocation:{' '}
            {formatPercentage(0)(
              params.targetAllocation.regularPortfolio.stocks
            )}
            , Spending Tilt: {format(params.scheduledWithdrawalGrowthRate)}
          </h2>
        )
      }
      case 'current-portfolio-value': {
        return <h2>{formatCurrency(params.savingsAtStartOfStartYear)}</h2>
      }
      case 'future-savings':
        return (
          <_EntriesSummary
            entries={params.savings}
            range={validYearRange(type)}
          />
        )
      case 'income-during-retirement':
        return (
          <_EntriesSummary
            entries={params.retirementIncome}
            range={validYearRange(type)}
          />
        )
      case 'extra-spending':
        const {fundedByBonds, fundedByRiskPortfolio} = params.withdrawals
        return (
          <>
            {fundedByBonds.length === 0 &&
              fundedByRiskPortfolio.length === 0 && <h2>None</h2>}
            {fundedByBonds.length > 0 && (
              <>
                <h2 className="mt-1 font-medium ">Essential</h2>
                <_EntriesSummary
                  entries={fundedByBonds}
                  range={validYearRange(type)}
                />
              </>
            )}
            {fundedByRiskPortfolio.length > 0 && (
              <>
                <h2 className="mt-1 font-medium ">Discretionary</h2>
                <_EntriesSummary
                  entries={fundedByRiskPortfolio}
                  range={validYearRange(type)}
                />
              </>
            )}
          </>
        )
      case 'spending-ceiling-and-floor': {
        return params.spendingCeiling === null &&
          params.spendingFloor === null ? (
          <h2>None</h2>
        ) : params.spendingCeiling === params.spendingFloor ? (
          <h2>
            Fixed Spending: {formatCurrency(fGet(params.spendingCeiling))}
          </h2>
        ) : (
          <h2>
            {params.spendingCeiling && (
              <span>Ceiling: {formatCurrency(params.spendingCeiling)}</span>
            )}
            {params.spendingFloor && (
              <span>Floor: {formatCurrency(params.spendingFloor)}</span>
            )}
          </h2>
        )
      }
      case 'legacy': {
        const {total, external} = params.legacy
        if (total === 0) {
          return <h2>None</h2>
        } else {
          return (
            <>
              <h2>Target: {formatCurrency(total)}. Real dollars.</h2>
              {external.map((x, i) => (
                <h2 key={i}>
                  {trimAndNullify(x.label) ?? '<no label>'}:{' '}
                  {formatCurrency(x.value)}.{' '}
                  {x.nominal ? 'Nominal dollars.' : 'Real dollars.'}
                </h2>
              ))}
              <h2>
                Stock Allocation:{' '}
                {formatPercentage(0)(
                  params.targetAllocation.legacyPortfolio.stocks
                )}
              </h2>
            </>
          )
        }
      }
      case 'expected-returns': {
        const format = formatPercentage(1)
        return (
          <h2>
            Stocks: {format(params.returns.expected.stocks)}, Bonds:{' '}
            {format(params.returns.expected.bonds)}
          </h2>
        )
      }
      case 'inflation': {
        const format = formatPercentage(1)
        return <h2>{format(params.inflation)}</h2>
      }
      default:
        noCase(type)
    }
  }
)

export const _EntriesSummary = React.memo(
  ({
    entries,
    range,
  }: {
    entries: ValueForYearRange[]
    range: {start: Year; end: Year}
  }) => {
    if (entries.length === 0) return <h2>None</h2>
    return (
      <ol className={`list-outside list-disc ml-3 grid gap-y-2 mt-1`}>
        {entries.map((x, i) => (
          <li key={i} className="">
            <ValueForYearRangeDisplay entry={x} range={range} />
          </li>
        ))}
      </ol>
    )
  }
)
