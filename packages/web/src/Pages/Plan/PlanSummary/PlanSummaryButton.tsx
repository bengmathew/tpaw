import { faCheck, faExclamation } from '@fortawesome/pro-solid-svg-icons'
import { faPlus } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  assert,
  assertFalse,
  fGet,
  getDefaultPlanParams,
  GlidePath,
  Person,
  RISK_TOLERANCE_VALUES,
  ValueForYearRange,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { PlanParamsExt } from '../../../TPAWSimulator/PlanParamsExt'
import {
  processExpectedReturns,
  processInflation,
} from '../../../TPAWSimulator/PlanParamsProcessed'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { Padding, paddingCSSStyle } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { trimAndNullify } from '../../../Utils/TrimAndNullify'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { noCase } from '../../../Utils/Utils'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { ChartUtils } from '../../Common/Chart/ChartUtils/ChartUtils'
import { ValueForYearRangeDisplay } from '../../Common/ValueForYearRangeDisplay'
import { useGetSectionURL } from '../Plan'
import { PlanInputModifiedBadge } from '../PlanInput/Helpers/PlanInputModifiedBadge'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import {
  isPlanSectionDialogInOverlayMode,
  planSectionDialogOrder,
} from '../PlanInput/Helpers/PlanSectionDialogPosition'
import { planSectionLabel } from '../PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { expectedReturnTypeLabel } from '../PlanInput/PlanInputExpectedReturns'
import { inflationTypeLabel } from '../PlanInput/PlanInputInflation'

type _Props = {
  padding: Padding
  type: PlanInputType
  section: PlanSectionName
  hideTitle?: boolean
  warn?: boolean
  flagAsModified?: boolean
  empty?: boolean
}
export const PlanSummaryButton = React.memo(
  React.forwardRef<HTMLButtonElement, _Props>(
    (
      {
        padding,
        type,
        section,
        warn = false,
        flagAsModified = false,
        empty = false,
        hideTitle = false,
      }: _Props,
      ref,
    ) => {
      const { params } = useSimulation()
      const getSectionURL = useGetSectionURL()
      const urlUpdater = useURLUpdater()
      const highlightColorDark = ChartUtils.color.gray[400]
      const isDisabled = useShouldDisablePlanSummaryButton(type)
      const highlightColor =
        section === type
          ? highlightColorDark
          : params.dialogPosition === type
          ? ChartUtils.color.orange[400]
          : ChartUtils.color.gray[100]

      if (hideTitle) assert(!warn)
      return (
        <button
          className={`block rounded-2xl  text-left w-full border-[2px] overflow-hidden ${
            isPlanSectionDialogInOverlayMode(params.dialogPosition)
              ? ''
              : 'disabled:opacity-20'
          } 
            ${params.dialogPosition === type ? 'bg-orange-50' : 'bg-cardBG'}`}
          ref={ref}
          style={{
            transitionProperty: 'border-color',
            transitionDuration:
              highlightColor === highlightColorDark ? '500ms' : '1250ms',
            borderColor: highlightColor,
          }}
          onClick={() => urlUpdater.push(getSectionURL(type))}
          disabled={isDisabled}
        >
          {empty ? (
            <div
              className={`relative  bg-gray-200/40  rounded-2xl -m-[2px] 
              ${
                params.dialogPosition === type
                  ? 'border-[3px] border-gray-200'
                  : 'border-[4px] border-dotted border-gray-400'
              }`}
              style={{ ...paddingCSSStyle(padding) }}
            >
              <div className=" flex items-center gap-x-2 mb-1">
                <FontAwesomeIcon className="text-3xl" icon={faPlus} />
                <h2 className="font-semibold mr-2 opacity-[0.75] ">
                  <span className="">Add {planSectionLabel(type)}</span>
                </h2>
              </div>
            </div>
          ) : (
            <div
              className="relative border-[3px] border-gray-200 rounded-2xl -m-[2px]"
              style={{ ...paddingCSSStyle(padding) }}
            >
              <PlanInputModifiedBadge show={flagAsModified} mainPage />
              {!hideTitle && (
                <div className=" flex  items-center mb-1">
                  <h2 className="font-semibold mr-2 flex">
                    <span className="">{planSectionLabel(type)}</span>
                  </h2>
                  {warn && (
                    <div className="h-[18px] w-[18px] flex items-center justify-center text-[13px] font-bold rounded-full bg-errorBlockBG text-errorBlockFG">
                      <FontAwesomeIcon icon={faExclamation} />
                    </div>
                  )}
                </div>
              )}
              <div className="grid gap-y-1 text-sm text-gray-600">
                <_SectionSummary type={type} />
              </div>
            </div>
          )}
        </button>
      )
    },
  ),
)

export const useShouldDisablePlanSummaryButton = (type: PlanInputType) => {
  const { dialogPosition } = useSimulation().params
  switch (type) {
    case 'age':
    case 'current-portfolio-balance':
    case 'future-savings':
    case 'income-during-retirement':
      return (
        planSectionDialogOrder.indexOf(dialogPosition) <
        planSectionDialogOrder.indexOf(type)
      )
    case 'extra-spending':
    case 'legacy':
    case 'risk':
    case 'spending-ceiling-and-floor':
    case 'inflation':
    case 'strategy':
    case 'expected-returns':
    case 'simulation':
    case 'dev':
      return dialogPosition !== 'done'
    default:
      noCase(type)
  }
}

const _SectionSummary = React.memo(({ type }: { type: PlanInputType }) => {
  const { params, paramsProcessed, paramsExt } = useSimulation()
  const marketData = useMarketData()
  const { validYearRange, pickPerson, yourOrYourPartners, withdrawalsStarted } =
    paramsExt
  const defaultParams = useMemo(() => getDefaultPlanParams(), [])
  switch (type) {
    case 'age': {
      if (params.dialogPosition === 'age') {
        return (
          <>
            {/* <h2>Set your current, retirement, and max age</h2> */}
            <h2>Current: </h2>
            <h2>Retirement: </h2>
            <h2>Max: </h2>
          </>
        )
      }
      if (params.people.withPartner) {
        const withdrawalPerson = pickPerson(params.people.withdrawalStart)
        const { person1, person2 } = params.people
        const forPerson = (person: Person) => (
          <>
            <h2 className="">Current: {person.ages.current}</h2>
            <h2 className="">
              Retirement:{' '}
              {person.ages.type === 'retired'
                ? '(retired)'
                : `${person.ages.retirement}`}
            </h2>
            <h2 className="text-start">Max: {person.ages.max}</h2>
          </>
        )
        return (
          <>
            <div
              className="grid gap-x-3 text-right"
              style={{ grid: 'auto/auto auto 1fr' }}
            >
              <h2 className="font-medium col-span-3 text-left">You</h2>
              {forPerson(person1)}

              <h2 className="mt-2 font-medium col-span-3 text-left">
                Your Partner
              </h2>
              {forPerson(person2)}
            </div>
            <h2 className="mt-1">
              Withdrawals start{' '}
              {withdrawalPerson.ages.type === 'retired'
                ? 'now.'
                : `at ${yourOrYourPartners(
                    params.people.withdrawalStart,
                  )} retirement.`}
            </h2>
          </>
        )
      } else {
        const { ages } = params.people.person1
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
      const { essential, discretionary } =
        params.adjustmentsToSpending.extraSpending
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
      const { total, external } =
        params.adjustmentsToSpending.tpawAndSPAW.legacy
      if (total === 0 && external.length === 0) {
        return <h2>None</h2>
      } else {
        return external.length === 0 ? (
          <h2>Target: {formatCurrency(total)} (real dollars)</h2>
        ) : (
          <>
            <div
              className="grid gap-x-2"
              style={{ grid: 'auto/1fr auto auto' }}
            >
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
                {formatCurrency(
                  paramsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy
                    .target,
                )}{' '}
              </h2>
              <h2 className="mt-2">(real dollars)</h2>
            </div>
          </>
        )
      }
    }
    case 'spending-ceiling-and-floor': {
      const { spendingCeiling } = params.adjustmentsToSpending.tpawAndSPAW
      const { spendingFloor } = params.adjustmentsToSpending.tpawAndSPAW
      return (
        <>
          <h2>
            Ceiling:{' '}
            {spendingCeiling ? formatCurrency(spendingCeiling) : 'None'}
          </h2>
          <h2>
            Floor: {spendingFloor ? formatCurrency(spendingFloor) : 'None'}
          </h2>
        </>
      )
    }

    case 'risk': {
      const { risk } = params
      const defaultRisk = defaultParams.risk
      switch (params.strategy) {
        case 'TPAW': {
          const advancedCount = _.filter([
            risk.tpaw.riskTolerance.deltaAtMaxAge !==
              defaultRisk.tpaw.riskTolerance.deltaAtMaxAge,
            risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 !==
              defaultRisk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
            risk.tpaw.timePreference !== defaultRisk.tpaw.timePreference,
          ]).length
          return (
            <>
              <h2>
                {`Risk Tolerance: ${risk.tpaw.riskTolerance.at20} (${fGet(
                  RISK_TOLERANCE_VALUES.SEGMENTS.find((x) =>
                    x.containsIndex(risk.tpaw.riskTolerance.at20),
                  ),
                ).label.toLowerCase()})`}
              </h2>
              {advancedCount > -0 && (
                <>
                  <h2 className="">Advanced</h2>
                  {defaultRisk.tpaw.riskTolerance.deltaAtMaxAge !==
                    risk.tpaw.riskTolerance.deltaAtMaxAge && (
                    <h2 className="ml-4">
                      Decrease Risk Tolerance With Age:{' '}
                      {-risk.tpaw.riskTolerance.deltaAtMaxAge}
                    </h2>
                  )}
                  {defaultRisk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 !==
                    risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 && (
                    <h2 className="ml-4">
                      Increase Risk Tolerance For Legacy:{' '}
                      {risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20}
                    </h2>
                  )}
                  {defaultRisk.tpaw.timePreference !==
                    risk.tpaw.timePreference && (
                    <h2 className="ml-4">
                      Spending Tilt:{' '}
                      {formatPercentage(1)(-risk.tpaw.timePreference)}
                    </h2>
                  )}
                </>
              )}
            </>
          )
        }
        case 'SPAW':
          return (
            <>
              <h2>Stock Allocation</h2>
              <div className="ml-4">
                <_GlidePath
                  className=""
                  glidePath={risk.spawAndSWR.allocation}
                  format={(x) => formatPercentage(0)(x)}
                />
              </div>
              <h2>
                Spending Tilt: {formatPercentage(1)(risk.spaw.spendingTilt)}
              </h2>
            </>
          )
        case 'SWR': {
          return (
            <>
              <h2>Stock Allocation</h2>
              <div className="ml-4">
                <_GlidePath
                  className=""
                  glidePath={risk.spawAndSWR.allocation}
                  format={(x) => formatPercentage(0)(x)}
                />
              </div>
              <h2>Withdrawal</h2>
              <h2 className="ml-4">
                {risk.swr.withdrawal.type === 'asPercent'
                  ? `${formatPercentage(1)(risk.swr.withdrawal.percent)} of ${
                      withdrawalsStarted
                        ? 'current portfolio balance'
                        : 'savings portfolio at retirement'
                    }`
                  : risk.swr.withdrawal.type === 'asAmount'
                  ? `${formatCurrency(risk.swr.withdrawal.amount)}`
                  : risk.swr.withdrawal.type === 'default'
                  ? // Default should have been changed to asPercent if we are showing this.
                    assertFalse()
                  : noCase(risk.swr.withdrawal)}
              </h2>
            </>
          )
        }
        default:
          noCase(params.strategy)
      }
    }
    case 'expected-returns': {
      const format = formatPercentage(1)
      const { stocks, bonds } = processExpectedReturns(
        params.returns.expected,
        marketData,
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
          {(['TPAW', 'SPAW', 'SWR'] as const).map((value) => (
            <_ChoiceItem
              key={value}
              value={value}
              selected={(x) => params.strategy === x}
              label={(x) => x}
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
  ({
    entries,
    range,
  }: {
    entries: ValueForYearRange[]
    range: SimpleRange
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
  },
)

const _GlidePathIntermediate = React.memo(
  ({
    intermediate,
    format,
  }: {
    intermediate: ReturnType<PlanParamsExt['glidePathIntermediateValidated']>
    format: (x: number) => string
  }) => {
    const { yearToStr } = useSimulation().paramsExt
    return (
      <>
        {intermediate.map((x) => (
          <>
            <h2 className={`${x.issue === 'none' ? '' : 'text-errorFG'}`}>
              {_.capitalize(yearToStr(x.year))}
            </h2>
            <h2 className="text-right">{format(x.stocks)}</h2>
          </>
        ))}
      </>
    )
  },
)

const _GlidePath = React.memo(
  ({
    className = '',
    glidePath,
    format,
  }: {
    className?: string
    glidePath: GlidePath
    format: (x: number) => string
  }) => {
    const { paramsExt } = useSimulation()
    const intermediate = paramsExt.glidePathIntermediateValidated(
      glidePath.intermediate,
    )

    return (
      <div
        className={`${className} inline-grid gap-x-10 items-center`}
        style={{ grid: 'auto/auto auto' }}
      >
        <_GlidePathIntermediate
          intermediate={intermediate.filter((x) => x.issue === 'before')}
          format={format}
        />
        <h2>Now</h2>
        <h2 className="text-right">{format(glidePath.start.stocks)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(
            (x) => x.issue !== 'before' && x.issue !== 'after',
          )}
          format={format}
        />
        <h2>Max Age</h2>
        <h2 className="text-right">{format(glidePath.end.stocks)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter((x) => x.issue === 'after')}
          format={format}
        />
      </div>
    )
  },
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
