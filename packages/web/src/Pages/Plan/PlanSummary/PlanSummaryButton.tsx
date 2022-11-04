import { faCheck, faExclamation } from '@fortawesome/pro-solid-svg-icons'
import { faPlus } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import Link from 'next/link'
import React from 'react'
import { riskLevelLabel } from '../../../TPAWSimulator/RiskLevelLabel'
import {
  GlidePath,
  Person,
  ValueForYearRange
} from '@tpaw/common'
import { PlanParamsExt } from '../../../TPAWSimulator/PlanParamsExt'
import {
  processExpectedReturns,
  processInflation
} from '../../../TPAWSimulator/PlanParamsProcessed'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { Padding, paddingCSSStyle } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { trimAndNullify } from '../../../Utils/TrimAndNullify'
import { assert, assertFalse, fGet, noCase } from '../../../Utils/Utils'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { ChartUtils } from '../../Common/Chart/ChartUtils/ChartUtils'
import { ValueForYearRangeDisplay } from '../../Common/ValueForYearRangeDisplay'
import { useGetSectionURL } from '../Plan'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import { planSectionLabel } from '../PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { expectedReturnTypeLabel } from '../PlanInput/PlanInputExpectedReturns'
import { inflationTypeLabel } from '../PlanInput/PlanInputInflation'

export const PlanSummaryButton = React.memo(
  ({
    padding,
    type,
    section,
    warn = false,
    flagAsModified = false,
    empty = false,
  }: {
    padding: Padding
    type: PlanInputType
    section: PlanSectionName
    warn?: boolean
    flagAsModified?: boolean
    empty?: boolean
  }) => {
    const {params} = useSimulation()
    const getSectionURL = useGetSectionURL()
    const highlightColorDark = ChartUtils.color.gray[400]
    const highlightColor =
      section === type ? highlightColorDark : ChartUtils.color.gray[100]

    return (
      <Link href={getSectionURL(type)} shallow>
        <a
          className={`block rounded-2xl bg-cardBG text-left w-full border-[2px] overflow-hidden `}
          style={{
            transitionProperty: 'border-color',
            transitionDuration:
              highlightColor === highlightColorDark ? '500ms' : '1250ms',
            borderColor: highlightColor,
          }}
        >
          {empty ? (
            <div
              className="relative border-[4px] border-dotted border-gray-400 bg-gray-200/40  rounded-2xl -m-[2px] "
              style={{...paddingCSSStyle(padding)}}
            >
              <div className=" flex items-center gap-x-2 mb-1">
                <FontAwesomeIcon className="text-3xl" icon={faPlus} />
                <h2 className="font-semibold mr-2 opacity-[0.75] ">
                  <span className="">
                    Add {planSectionLabel(type, params.strategy)}
                  </span>
                </h2>
              </div>
            </div>
          ) : (
            <div
              className="relative border-[3px] border-gray-200 rounded-2xl -m-[2px]"
              style={{...paddingCSSStyle(padding)}}
            >
              {flagAsModified && (
                <h2 className="absolute right-2 text-xs top-1.5 border border-gray-200 bg-gray-100 px-2 rounded-lg">
                  modified
                </h2>
              )}
              <div className=" flex  items-center mb-1">
                <h2 className="font-semibold mr-2 flex">
                  <span className="">
                    {planSectionLabel(type, params.strategy)}
                  </span>
                </h2>
                {warn && (
                  <div className="h-[18px] w-[18px] flex items-center justify-center text-[13px] font-bold rounded-full bg-errorBlockBG text-errorBlockFG">
                    <FontAwesomeIcon icon={faExclamation} />
                  </div>
                )}
              </div>
              <div className="grid gap-y-1 text-sm text-gray-600">
                <_SectionSummary type={type} />
              </div>
            </div>
          )}
        </a>
      </Link>
    )
  }
)

const _SectionSummary = React.memo(({type}: {type: PlanInputType}) => {
  const {params, paramsProcessed, paramsExt} = useSimulation()
  const marketData = useMarketData()
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
      return <h2>{formatCurrency(params.currentPortfolioBalance)}</h2>
    }
    case 'future-savings':
      return (
        <_EntriesSummary
          entries={params.futureSavings}
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
      const {essential, discretionary} = params.extraSpending
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
      const {total, external} = params.legacy.tpawAndSPAW
      if (total === 0 && external.length === 0) {
        return <h2>None</h2>
      } else {
        return external.length === 0 ? (
          <h2>Target: {formatCurrency(total)} (real dollars)</h2>
        ) : (
          <>
            <div className="grid gap-x-2" style={{grid: 'auto/1fr auto auto'}}>
              <h2 className="mt-2">Total Target</h2>
              <h2 className="text-right mt-2">{formatCurrency(total)}</h2>
              <h2 className="mt-2">(real dollars)</h2>
              <h2 className=" col-span-3 mt-2">Non-portfolio Sources</h2>
              {external.map((x, i) => (
                <React.Fragment key={i}>
                  <h2 className="ml-4 mt-1">
                    {trimAndNullify(x.label) ?? '<no label>'}
                  </h2>
                  <h2 className="mt-1 text-right">
                    {formatCurrency(x.value)}{' '}
                  </h2>
                  <h2 className="mt-1">
                    {x.nominal ? '(nominal dollars)' : '(real dollars)'}{' '}
                  </h2>
                </React.Fragment>
              ))}
              <h2 className="mt-2">Remaining Target</h2>
              <h2 className="mt-2 text-right">
                {formatCurrency(paramsProcessed.legacy.tpawAndSPAW.target)}{' '}
              </h2>
              <h2 className="mt-2">(real dollars)</h2>
            </div>
          </>
        )
      }
    }
    case 'risk-level': {
      assert(params.risk.useTPAWPreset)
      return (
          <>
            {(
              [
                'riskLevel-1',
                'riskLevel-2',
                'riskLevel-3',
                'riskLevel-4',
              ] as const
            ).map(
              value =>
                value && (
                  <_ChoiceItem
                    key={value}
                    value={value}
                    label={riskLevelLabel}
                    selected={value => params.risk.tpawPreset === value}
                  />
                )
            )}
          </>
      )
    }
    case 'stock-allocation': {
      assert(!params.risk.useTPAWPreset)
      return (
        <>
          {params.strategy === 'TPAW' ? (
            <>
              <h2 className="font-medium mt-2">For Retirement Spending</h2>
              <_GlidePath
                className="ml-4"
                glidePath={params.risk.tpaw.allocation}
              />
              <h2 className="font-medium mt-2">For Legacy</h2>
              <div className={`ml-4 flex justify-between`}>
                <h2 className="">Stock Allocation: </h2>
                <h2 className="text-right">
                  {formatPercentage(0)(
                    params.risk.tpaw.allocationForLegacy.stocks
                  )}
                </h2>
              </div>
            </>
          ) : (
            <>
              <_GlidePath
                className=""
                glidePath={params.risk.spawAndSWR.allocation}
              />
            </>
          )}
        </>
      )
    }
    case 'spending-tilt': {
      assert(!params.risk.useTPAWPreset)
      return (
        <h2>{formatPercentage(1)(params.risk.tpawAndSPAW.spendingTilt)}</h2>
      )
    }
    case 'spending-ceiling-and-floor': {
      assert(!params.risk.useTPAWPreset)
      const {spendingCeiling, spendingFloor} = params.risk.tpawAndSPAW
      return spendingCeiling === null && spendingFloor === null ? (
        <h2>None</h2>
      ) : spendingCeiling === spendingFloor ? (
        <h2>Fixed Spending: {formatCurrency(fGet(spendingCeiling))}</h2>
      ) : (
        <>
          {spendingCeiling && (
            <h2>Ceiling: {formatCurrency(spendingCeiling)}</h2>
          )}
          {spendingFloor && <h2>Floor: {formatCurrency(spendingFloor)}</h2>}
        </>
      )
    }
    case 'lmp': {
      assert(!params.risk.useTPAWPreset)
      return (
        <h2>
          {params.risk.tpawAndSPAW.lmp === 0
            ? 'None'
            : formatCurrency(params.risk.tpawAndSPAW.lmp)}
        </h2>
      )
    }
    case 'withdrawal': {
      const {withdrawalsStarted} = paramsExt
      const {withdrawal} = params.risk.swr
      return (
        <h2>
          {withdrawal.type === 'asPercent'
            ? `${formatPercentage(1)(withdrawal.percent)} of ${
                withdrawalsStarted
                  ? 'current portfolio balance'
                  : 'savings portfolio at retirement'
              }`
            : withdrawal.type === 'asAmount'
            ? `${formatCurrency(withdrawal.amount)}`
            : withdrawal.type === 'default'
            ? // Default should have been changed to asPercent if we are showing this.
              assertFalse()
            : noCase(withdrawal)}
        </h2>
      )
    }
    case 'expected-returns': {
      const format = formatPercentage(1)
      const {stocks, bonds} = processExpectedReturns(
        params.returns.expected,
        marketData
      )
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
      return (
        <h2>
          {inflationTypeLabel(params.inflation)}:{' '}
          {format(processInflation(params.inflation, marketData))}
        </h2>
      )
    }
    case 'strategy': {
      return (
        <>
          {(['TPAW', 'SPAW', 'SWR'] as const).map(value => (
            <_ChoiceItem
              key={value}
              value={value}
              selected={x => params.strategy === x}
              label={x => x}
            />
          ))}
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
    intermediate: ReturnType<PlanParamsExt['glidePathIntermediateValidated']>
  }) => {
    const {yearToStr} = useSimulation().paramsExt
    return (
      <>
        {intermediate.map(x => (
          <>
            <h2 className={`${x.issue === 'none' ? '' : 'text-errorFG'}`}>
              {_.capitalize(yearToStr(x.year))}
            </h2>
            <h2 className="text-right">{formatPercentage(0)(x.stocks)}</h2>
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
        <h2>Now</h2>
        <h2 className="text-right">
          {formatPercentage(0)(glidePath.start.stocks)}
        </h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(
            x => x.issue !== 'before' && x.issue !== 'after'
          )}
        />
        <h2>Max Age</h2>
        <h2 className="text-right">
          {formatPercentage(0)(glidePath.end.stocks)}
        </h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(x => x.issue === 'after')}
        />
      </div>
    )
  }
)

function _ChoiceItem<Value>({
  value,
  label,
  selected,
}: {
  value: Value
  label: (value: Value) => string
  selected: (value: Value) => boolean
}) {
  return (
    <h2 className={selected(value) ? '' : 'lighten-2'}>
      {label(value)}
      {selected(value) && <FontAwesomeIcon className="ml-2" icon={faCheck} />}
    </h2>
  )
}
