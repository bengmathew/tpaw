import {faExclamation} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap, Power1} from 'gsap'
import _ from 'lodash'
import React, {useEffect, useRef} from 'react'
import {
  GlidePath,
  Person,
  ValueForYearRange,
} from '../../../TPAWSimulator/TPAWParams'
import {TPAWParamsExt} from '../../../TPAWSimulator/TPAWParamsExt'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {Padding, paddingCSS} from '../../../Utils/Geometry'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {trimAndNullify} from '../../../Utils/TrimAndNullify'
import {fGet, noCase} from '../../../Utils/Utils'
import {Footer} from '../../App/Footer'
import {useSimulation} from '../../App/WithSimulation'
import {ChartUtils} from '../../Common/Chart/ChartUtils/ChartUtils'
import {ValueForYearRangeDisplay} from '../../Common/ValueForYearRangeDisplay'
import {analyzeYearsInParams} from './Helpers/AnalyzeYearsInParams'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {ParamsInputType} from './Helpers/ParamsInputType'

export const ParamsInputSummary = React.memo(
  ({
    layout,
    state,
    setState,
    cardPadding,
  }: {
    layout: 'mobile' | 'desktop' | 'laptop'
    state: ParamsInputType | 'summary'
    setState: (state: ParamsInputType) => void
    cardPadding: Padding
  }) => {
    const {params, paramsExt} = useSimulation()
    const {asYFN, withdrawalStartYear} = paramsExt
    const isRetired = asYFN(withdrawalStartYear) <= 0

    const labelStyle: React.CSSProperties = {
      paddingLeft: `${cardPadding.left}px`,
      paddingRight: `${cardPadding.right}px`,
    }

    return (
      <div
        className={` grid w-full`} // This is needed if we start in exited state.
        style={{grid: '1fr auto/1fr'}}
      >
        <div
          className={`flex flex-col items-start mb-16 ${
            layout === 'desktop' ? 'max-w-[500px]' : ''
          }`}
        >
          <div
            className={`flex flex-col gap-y-12 sm:gap-y-16 relative z-0 w-full`}
          >
            <div className="">
              <h2
                className={`text-[18px] sm:text-xl font-bold mb-4`}
                style={labelStyle}
              >
                Age and Retirement
              </h2>
              <div className="flex flex-col gap-y-4">
                <_Button
                  type="age-and-retirement"
                  setState={setState}
                  state={state}
                  showHeading={false}
                  padding={cardPadding}
                />
              </div>
            </div>
            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={labelStyle}
              >
                Savings and Income
              </h2>
              <div className="flex flex-col gap-y-4 ">
                <_Button
                  type="current-portfolio-balance"
                  setState={setState}
                  state={state}
                  showHeading
                  padding={cardPadding}
                />
                {!isRetired && (
                  <_Button
                    type="future-savings"
                    setState={setState}
                    state={state}
                    showHeading
                    warn={!_paramsOk(paramsExt, 'future-savings')}
                    padding={cardPadding}
                  />
                )}
                <_Button
                  type="income-during-retirement"
                  setState={setState}
                  state={state}
                  showHeading
                  warn={!_paramsOk(paramsExt, 'income-during-retirement')}
                  padding={cardPadding}
                />
              </div>
            </div>
            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={labelStyle}
              >
                Spending
              </h2>
              <div className="flex flex-col gap-y-4">
                <_Button
                  type="extra-spending"
                  setState={setState}
                  state={state}
                  showHeading
                  warn={!_paramsOk(paramsExt, 'extra-spending')}
                  padding={cardPadding}
                />
                <_Button
                  type="spending-ceiling-and-floor"
                  setState={setState}
                  state={state}
                  showHeading
                  padding={cardPadding}
                />
                <_Button
                  type="legacy"
                  setState={setState}
                  state={state}
                  showHeading
                  padding={cardPadding}
                />
              </div>
            </div>

            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={labelStyle}
              >
                Risk and Time Preference
              </h2>
              <div className="flex flex-col gap-y-4">
                <_Button
                  type="risk-and-time-preference"
                  setState={setState}
                  state={state}
                  showHeading={false}
                  padding={cardPadding}
                />
              </div>
            </div>

            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={labelStyle}
              >
                {paramsInputLabel('strategy')}
              </h2>
              <div className="flex flex-col gap-y-4">
                <_Button
                  type="strategy"
                  setState={setState}
                  state={state}
                  showHeading={false}
                  padding={cardPadding}
                  warn={!_paramsOk(paramsExt, 'strategy')}
                />
              </div>
            </div>

            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={labelStyle}
              >
                Expected Returns and Inflation
              </h2>
              <div className="flex flex-col gap-y-4">
                <_Button
                  type="expected-returns"
                  setState={setState}
                  state={state}
                  showHeading
                  padding={cardPadding}
                />
                <_Button
                  type="inflation"
                  setState={setState}
                  state={state}
                  showHeading
                  padding={cardPadding}
                />
              </div>
            </div>
          </div>
        </div>
        {layout !== 'laptop' && <Footer />}
      </div>
    )
  }
)

const _Button = React.memo(
  ({
    padding,
    type,
    setState,
    warn = false,
    state,
    showHeading,
  }: {
    padding: Padding
    type: ParamsInputType
    state: ParamsInputType | 'summary'
    setState: (state: ParamsInputType) => void
    warn?: boolean
    showHeading: boolean
  }) => {
    const ref = useRef<HTMLButtonElement | null>(null)
    const highlightColorDark = ChartUtils.color.gray[400]
    const highlightColor =
      state === type ? highlightColorDark : ChartUtils.color.gray[100]
    useEffect(() => {
      const tween = gsap.to(ref.current, {
        borderColor: highlightColor,
        duration: highlightColor === highlightColorDark ? 0.5 : 1.25,
        ease: Power1.easeIn,
      })
      return () => {
        tween.kill()
      }
    }, [highlightColor, highlightColorDark])

    return (
      <button
        className={` rounded-2xl text-left w-full   border-planBG border-[2px] overflow-hidden`}
        onClick={() => setState(type)}
        ref={ref}
      >
        <div className="border-[3px] bg-cardBG  border-gray-00 rounded-2xl -m-[2px] p-[2px]">
          <div className="" style={{padding: paddingCSS(padding)}}>
            <div className=" flex  items-center mb-1">
              {showHeading && (
                <h2 className="font-semibold mr-2">{paramsInputLabel(type)}</h2>
              )}
              {warn && (
                <h2 className="h-[18px] w-[18px] flex items-center justify-center text-[13px] font-bold rounded-full bg-errorBlockBG text-errorBlockFG">
                  <FontAwesomeIcon icon={faExclamation} />
                </h2>
              )}
            </div>
            <div className="grid gap-y-1 text-sm text-gray-600">
              <_SectionSummary type={type} />
            </div>
          </div>
        </div>
      </button>
    )
  }
)

export const _SectionSummary = React.memo(
  ({className = '', type}: {className?: string; type: ParamsInputType}) => {
    const {params, paramsProcessed, paramsExt} = useSimulation()
    const {validYearRange, pickPerson, yourOrYourPartners, yearToStr} =
      paramsExt

    switch (type) {
      case 'age-and-retirement': {
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
          const {ages} = params.people.person1
          return ages.type === 'retired' ? (
            <>
              <h2>Retired</h2>
              <h2>Current: {ages.current}</h2>
              <h2>Max: {ages.max}</h2>
            </>
          ) : (
            <>
              <h2>Current: {ages.current}</h2>
              <h2>Retirement: {ages.retirement}</h2>
              <h2>Max: {ages.max}</h2>
            </>
          )
        }
      }
      case 'risk-and-time-preference': {
        const format = formatPercentage(1)
        return (
          <>
            {params.strategy === 'TPAW' && (
              <h2>
                Stock Allocation:{' '}
                {formatPercentage(0)(
                  params.targetAllocation.regularPortfolio.forTPAW.stocks
                )}
              </h2>
            )}
            <h2>
              Spending Tilt: {format(params.scheduledWithdrawalGrowthRate)}
            </h2>
          </>
        )
      }
      case 'current-portfolio-balance': {
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
        const {essential, discretionary} = params.withdrawals
        return (
          <>
            {essential.length === 0 &&
              discretionary.length === 0 && <h2>None</h2>}
            {essential.length > 0 && (
              <>
                <h2 className="mt-1 font-medium ">Essential</h2>
                <_EntriesSummary
                  entries={essential}
                  range={validYearRange(type)}
                />
              </>
            )}
            {discretionary.length > 0 && (
              <>
                <h2 className="mt-1 font-medium ">Discretionary</h2>
                <_EntriesSummary
                  entries={discretionary}
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
          <>
            {params.spendingCeiling && (
              <h2>Ceiling: {formatCurrency(params.spendingCeiling)}</h2>
            )}
            {params.spendingFloor && (
              <h2>Floor: {formatCurrency(params.spendingFloor)}</h2>
            )}
          </>
        )
      }
      case 'legacy': {
        const {total, external} = params.legacy
        if (total === 0 && external.length === 0) {
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
              {external.length > 0 && (
                <h2>
                  Remaining: {formatCurrency(paramsProcessed.legacy.target)}.
                  Real dollars.
                </h2>
              )}
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
          <>
            <h2>Stocks: {format(params.returns.expected.stocks)}</h2>
            <h2>Bonds: {format(params.returns.expected.bonds)}</h2>
          </>
        )
      }
      case 'inflation': {
        const format = formatPercentage(1)
        return <h2>{format(params.inflation)}</h2>
      }
      case 'strategy': {
        switch (params.strategy) {
          case 'TPAW':
            return (
              <h2 className="flex items-center">Total portfolio approach</h2>
            )
          case 'SPAW':
            return (
              <>
                <h2 className="flex items-center">
                  Savings portfolio approach with the following stock
                  allocation:
                </h2>
                <div className="">
                  {/* <h2 className="">Stock Allocation:</h2> */}
                  <_GlidePath
                    className=""
                    glidePath={params.targetAllocation.regularPortfolio.forSPAW}
                  />
                </div>
              </>
            )

          default:
            noCase(params.strategy)
        }
      }
      default:
        noCase(type)
    }
  }
)
1
export const _EntriesSummary = React.memo(
  ({entries, range}: {entries: ValueForYearRange[]; range: SimpleRange}) => {
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

type _Type =
  | 'future-savings'
  | 'income-during-retirement'
  | 'extra-spending'
  | 'strategy'
export const _paramsOk = (paramsExt: TPAWParamsExt, type: _Type) => {
  const {valueForYearRange, glidePath} = analyzeYearsInParams(paramsExt)
  return (
    valueForYearRange
      .filter(x => x.section === type)
      .every(x => x.boundsCheck.start === 'ok' && x.boundsCheck.end === 'ok') &&
    glidePath
      .filter(x => x.section === type)
      .every(x => x.analyzed.every(x => x.issue === 'none'))
  )
}

const _GlidePathIntermediate = React.memo(
  ({
    intermediate,
  }: {
    intermediate: ReturnType<TPAWParamsExt['glidePathIntermediateValidated']>
  }) => {
    const {yearToStr} = useSimulation().paramsExt
    return (
      <>
        {intermediate.map(x => (
          <>
            <h2 className={`${x.issue === 'none' ? '' : 'text-errorFG'}`}>
              {_.capitalize(yearToStr(x.year))}
            </h2>
            <h2>{formatPercentage(0)(x.stocks)}</h2>
          </>
        ))}
      </>
    )
  }
)

const _GlidePath = React.memo(
  ({className = '', glidePath}: {className?: string; glidePath: GlidePath}) => {
    const {paramsExt} = useSimulation()
    const intermediate = paramsExt.glidePathIntermediateValidated(
      glidePath.intermediate
    )
    return (
      <div
        className={`${className} inline-grid gap-x-10 items-center`}
        style={{grid: 'auto/auto auto'}}
      >
        <_GlidePathIntermediate
          intermediate={intermediate.filter(x => x.issue === 'before')}
        />
        <h2>Start</h2>
        <h2>{formatPercentage(0)(glidePath.start.stocks)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(
            x => x.issue !== 'before' && x.issue !== 'after'
          )}
        />
        <h2>End</h2>
        <h2>{formatPercentage(0)(glidePath.end.stocks)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(x => x.issue === 'after')}
        />
      </div>
    )
  }
)
