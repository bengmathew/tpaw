import { faCircle, faExclamation } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { gsap, Power1 } from 'gsap'
import _ from 'lodash'
import React, { useEffect, useRef } from 'react'
import {
  GlidePath,
  Person,
  TPAWParams,
  ValueForYearRange
} from '../../../../TPAWSimulator/TPAWParams'
import { TPAWParamsExt } from '../../../../TPAWSimulator/TPAWParamsExt'
import { processExpectedReturns } from '../../../../TPAWSimulator/TPAWParamsProcessed'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { Padding, paddingCSS } from '../../../../Utils/Geometry'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { trimAndNullify } from '../../../../Utils/TrimAndNullify'
import { assertFalse, fGet, noCase } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { ValueForYearRangeDisplay } from '../../../Common/ValueForYearRangeDisplay'
import { paramsInputLabel } from '../Helpers/ParamsInputLabel'
import { ParamsInputType } from '../Helpers/ParamsInputType'
import { expectedReturnTypeLabel } from '../ParamsInputExpectedReturns'

export const ParamsInputSummaryButton = React.memo(
  ({
    padding,
    type,
    state,
    setState,
    warn = false,
    strategies = ['TPAW', 'SPAW', 'SWR'],
    flagAsModified = false,
  }: {
    padding: Padding
    type: ParamsInputType
    state: ParamsInputType | 'summary'
    setState: (state: ParamsInputType) => void
    warn?: boolean
    strategies?: TPAWParams['strategy'][]
    flagAsModified?: boolean
  }) => {
    const {params} = useSimulation()
    const shouldShow = strategies.includes(params.strategy)
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

    if (!shouldShow) return <></>
    return (
      <button
        className={` rounded-2xl text-left w-full   border-planBG border-[2px] overflow-hidden disabled:lighten-2`}
        onClick={() => setState(type)}
        ref={ref}
      >
        <div className="border-[3px] bg-cardBG  border-gray-00 rounded-2xl -m-[2px] p-[2px]">
          <div className="relative" style={{padding: paddingCSS(padding)}}>
            {flagAsModified && (
              <h2 className="absolute right-2 text-xs top-1.5 border border-gray-200 bg-gray-100 px-2 rounded-lg">
                modified
              </h2>
            )}
            <div className=" flex  items-center mb-1">
              <h2 className="font-semibold mr-2 flex">
                <span className="">{paramsInputLabel(type)}</span>
              </h2>
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

const _SectionSummary = React.memo(({type}: {type: ParamsInputType}) => {
  const {params, paramsProcessed, paramsExt} = useSimulation()
  const {validYearRange, pickPerson, yourOrYourPartners} = paramsExt

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
    case 'extra-spending': {
      const {essential, discretionary} = params.withdrawals
      const showLabels = params.strategy !== 'SWR'
      return (
        <>
          {essential.length === 0 && discretionary.length === 0 && (
            <h2>None</h2>
          )}
          {essential.length > 0 && (
            <>
              {showLabels && <h2 className="mt-1 font-medium ">Essential</h2>}
              <_EntriesSummary
                entries={essential}
                range={validYearRange(type)}
              />
            </>
          )}
          {discretionary.length > 0 && (
            <>
              {showLabels && (
                <h2 className="mt-1 font-medium ">Discretionary</h2>
              )}
              <_EntriesSummary
                entries={discretionary}
                range={validYearRange(type)}
              />
            </>
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
                Remaining: {formatCurrency(paramsProcessed.legacy.target)}. Real
                dollars.
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
    case 'stock-allocation': {
      return (
        <>
          {params.strategy === 'TPAW' ? (
            <h2>
              On Total Portfolio:{' '}
              {formatPercentage(0)(
                params.targetAllocation.regularPortfolio.forTPAW.stocks
              )}
            </h2>
          ) : (
            <div className="">
              <h2>On Savings Portfolio:</h2>
              <_GlidePath
                className=""
                glidePath={
                  params.targetAllocation.regularPortfolio.forSPAWAndSWR
                }
              />
            </div>
          )}
        </>
      )
    }
    case 'spending-tilt': {
      return (
        <h2>{formatPercentage(1)(params.scheduledWithdrawalGrowthRate)}</h2>
      )
    }
    case 'spending-ceiling-and-floor': {
      return params.spendingCeiling === null &&
        params.spendingFloor === null ? (
        <h2>None</h2>
      ) : params.spendingCeiling === params.spendingFloor ? (
        <h2>Fixed Spending: {formatCurrency(fGet(params.spendingCeiling))}</h2>
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
    case 'lmp': {
      return (
        <h2>
          {params.withdrawals.lmp === 0
            ? 'None'
            : formatCurrency(params.withdrawals.lmp)}
        </h2>
      )
    }
    case 'withdrawal': {
      const {withdrawalsStarted} = paramsExt
      return (
        <h2>
          {params.swrWithdrawal.type === 'asPercent'
            ? `${formatPercentage(1)(params.swrWithdrawal.percent)} of ${
                withdrawalsStarted
                  ? 'current portfolio balance'
                  : 'savings portfolio at retirement'
              }`
            : params.swrWithdrawal.type === 'asAmount'
            ? `${formatCurrency(params.swrWithdrawal.amount)}`
            : params.swrWithdrawal.type === 'default'
            ? // Default should have been changed to asPercent if we are showing this.
              assertFalse()
            : noCase(params.swrWithdrawal)}
        </h2>
      )
    }
    case 'expected-returns': {
      const format = formatPercentage(1)
      const {stocks, bonds} = processExpectedReturns(params.returns.expected)
      return (
        <>
          <h2>{expectedReturnTypeLabel(params.returns.expected)}</h2>
          <h2>Stocks: {format(stocks)}</h2>
          <h2>Bonds: {format(bonds)}</h2>
        </>
      )
    }
    case 'inflation': {
      const format = formatPercentage(1)
      return <h2>{format(params.inflation)}</h2>
    }
    case 'compare-strategies': {
      return (
        <>
          <h2 className="flex items-center gap-x-3">
            <span
              className={`${params.strategy === 'TPAW' ? 'font-bold' : ''}`}
            >
              TPAW{' '}
            </span>
            <FontAwesomeIcon className="text-[4px] lighten" icon={faCircle} />
            <span
              className={`${params.strategy === 'SPAW' ? 'font-bold' : ''}`}
            >
              SPAW
            </span>
            <FontAwesomeIcon className="text-[4px] lighten" icon={faCircle} />
            <span className={`${params.strategy === 'SWR' ? 'font-bold' : ''}`}>
              SWR
            </span>
          </h2>
        </>
      )
    }
    case 'simulation': {
      return (
        <>
          <h2>
            {params.sampling === 'monteCarlo'
              ? 'Monte Carlo sequence'
              : 'Historical sequence'}
          </h2>
        </>
      )
    }
    case 'dev':
      if (
        params.returns.historical.type === 'default' &&
        !params.display.alwaysShowAllYears
      )
        return <h2>None</h2>
      return (
        <div>
          {params.returns.historical.type === 'default' && (
            <h2>Real Historical Returns</h2>
          )}
          {params.returns.historical.type === 'fixed' && (
            <>
              <h2>Fixed Historical Returns</h2>
              <h2 className="ml-4">
                Stocks: {formatPercentage(1)(params.returns.historical.stocks)}
              </h2>
              <h2 className="ml-4">
                Bonds: {formatPercentage(1)(params.returns.historical.bonds)}
              </h2>
            </>
          )}
          {params.display.alwaysShowAllYears && <h2>Show all years</h2>}
        </div>
      )
    default:
      noCase(type)
  }
})

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
