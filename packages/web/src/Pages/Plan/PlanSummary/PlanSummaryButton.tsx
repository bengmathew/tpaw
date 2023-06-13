import { faCheck, faExclamation } from '@fortawesome/pro-solid-svg-icons'
import { faPlus } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  EXPECTED_ANNUAL_RETURN_PRESETS,
  GlidePath,
  RISK_TOLERANCE_VALUES,
  SUGGESTED_ANNUAL_INFLATION,
  ValueForMonthRange,
  assertFalse,
  fGet,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { ParamsExtended } from '../../../TPAWSimulator/ExtentParams'
import { normalizeGlidePath } from '../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessRisk'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { Padding, paddingCSSStyle } from '../../../Utils/Geometry'
import { numMonthsStr } from '../../../Utils/NumMonthsStr'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { trimAndNullify } from '../../../Utils/TrimAndNullify'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { noCase } from '../../../Utils/Utils'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { ChartUtils } from '../../Common/Chart/ChartUtils/ChartUtils'
import { monthToStringForGlidePath } from '../../Common/Inputs/GlidePathInput'
import { ValueForMonthRangeDisplay } from '../../Common/ValueForMonthRangeDisplay'
import { PlanInputAgeSummary } from '../../Print/PrintInputSection'
import { useGetSectionURL } from '../Plan'
import { PlanInputModifiedBadge } from '../PlanInput/Helpers/PlanInputModifiedBadge'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import {
  isPlanSectionDialogInOverlayMode,
  planSectionDialogOrder,
} from '../PlanInput/Helpers/PlanSectionDialogPosition'
import { planSectionLabel } from '../PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { PlanInputCurrentPortfolioBalanceSummary } from '../PlanInput/PlanInputCurrentPortfolioBalance'
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
          : params.plan.dialogPosition === type
          ? ChartUtils.color.orange[400]
          : ChartUtils.color.gray[100]

      return (
        <button
          className={`block rounded-2xl  text-left w-full border-[2px] overflow-hidden ${
            isPlanSectionDialogInOverlayMode(params.plan.dialogPosition)
              ? ''
              : 'disabled:opacity-20'
          } 
            ${
              params.plan.dialogPosition === type ? 'bg-orange-50' : 'bg-cardBG'
            }`}
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
                params.plan.dialogPosition === type
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
              {(!hideTitle || warn) && (
                <div className="flex  items-center mb-1">
                  {!hideTitle && (
                    <h2 className="font-semibold mr-2 flex">
                      <span className="">{planSectionLabel(type)}</span>
                    </h2>
                  )}
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
  const { dialogPosition } = useSimulation().params.plan
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
  const { params, paramsProcessed, paramsExt, defaultParams } = useSimulation()
  const marketData = useMarketData()
  const { validMonthRangeAsMFN, withdrawalsStarted } = paramsExt
  switch (type) {
    case 'age': {
      return <PlanInputAgeSummary />
    }
    case 'current-portfolio-balance': {
      return <PlanInputCurrentPortfolioBalanceSummary />
    }
    case 'future-savings':
      return (
        <PlanInputValueForMonthRangeSummary
          entries={params.plan.wealth.futureSavings}
          range={validMonthRangeAsMFN(type)}
        />
      )
    case 'income-during-retirement':
      return (
        <PlanInputValueForMonthRangeSummary
          entries={params.plan.wealth.retirementIncome}
          range={validMonthRangeAsMFN(type)}
        />
      )
    case 'extra-spending': {
      const { essential, discretionary } =
        params.plan.adjustmentsToSpending.extraSpending
      const showLabels = params.plan.advanced.strategy !== 'SWR'
      return (
        <>
          {essential.length === 0 && discretionary.length === 0 && (
            <h2>None</h2>
          )}
          {essential.length > 0 && (
            <>
              {showLabels && <h2 className="mt-1 font-medium ">Essential</h2>}
              <PlanInputValueForMonthRangeSummary
                entries={essential}
                range={validMonthRangeAsMFN(type)}
              />
            </>
          )}
          {discretionary.length > 0 && (
            <>
              {showLabels && (
                <h2 className="mt-1 font-medium ">Discretionary</h2>
              )}
              <PlanInputValueForMonthRangeSummary
                entries={discretionary}
                range={validMonthRangeAsMFN(type)}
              />
            </>
          )}
        </>
      )
    }
    case 'legacy': {
      const { total, external } =
        params.plan.adjustmentsToSpending.tpawAndSPAW.legacy
      return external.length === 0 ? (
        <h2>Target: {formatCurrency(total)} (real dollars)</h2>
      ) : (
        <>
          <div className="grid gap-x-2" style={{ grid: 'auto/1fr auto auto' }}>
            <h2 className="mt-2">Total Target</h2>
            <h2 className="text-right mt-2">{formatCurrency(total)}</h2>
            <h2 className="mt-2">(real dollars)</h2>
            <h2 className=" col-span-3 mt-2">Non-portfolio Sources</h2>
            {external.map((x, i) => (
              <React.Fragment key={i}>
                <h2 className="ml-4 mt-1">
                  {trimAndNullify(x.label) ?? '<no label>'}
                </h2>
                <h2 className="mt-1 text-right">{formatCurrency(x.value)} </h2>
                <h2 className="mt-1">
                  {x.nominal ? '(nominal dollars)' : '(real dollars)'}{' '}
                </h2>
              </React.Fragment>
            ))}
            <h2 className="mt-2">Remaining Target</h2>
            <h2 className="mt-2 text-right">
              {formatCurrency(
                paramsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.target,
              )}{' '}
            </h2>
            <h2 className="mt-2">(real dollars)</h2>
          </div>
        </>
      )
    }
    case 'spending-ceiling-and-floor': {
      const { monthlySpendingCeiling, monthlySpendingFloor } =
        params.plan.adjustmentsToSpending.tpawAndSPAW
      return (
        <>
          <h2>
            Ceiling:{' '}
            {monthlySpendingCeiling
              ? formatCurrency(monthlySpendingCeiling)
              : 'None'}
          </h2>
          <h2>
            Floor:{' '}
            {monthlySpendingFloor
              ? formatCurrency(monthlySpendingFloor)
              : 'None'}
          </h2>
        </>
      )
    }

    case 'risk': {
      const { risk } = params.plan
      const defaultRisk = defaultParams.plan.risk
      switch (params.plan.advanced.strategy) {
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
                <PlanInputGlidePathSummary
                  className=""
                  glidePath={risk.spawAndSWR.allocation}
                  format={(x) => formatPercentage(0)(x)}
                />
              </div>
              <h2>
                Spending Tilt:{' '}
                {formatPercentage(1)(risk.spaw.annualSpendingTilt)}
              </h2>
            </>
          )
        case 'SWR': {
          return (
            <>
              <h2>Stock Allocation</h2>
              <div className="ml-4">
                <PlanInputGlidePathSummary
                  className=""
                  glidePath={risk.spawAndSWR.allocation}
                  format={(x) => formatPercentage(0)(x)}
                />
              </div>
              <h2>Withdrawal</h2>
              <h2 className="ml-4">
                {risk.swr.withdrawal.type === 'asPercentPerYear'
                  ? `${formatPercentage(1)(
                      risk.swr.withdrawal.percentPerYear,
                    )} of ${
                      withdrawalsStarted
                        ? 'current portfolio balance'
                        : 'savings portfolio at retirement'
                    }`
                  : risk.swr.withdrawal.type === 'asAmountPerMonth'
                  ? `${formatCurrency(
                      risk.swr.withdrawal.amountPerMonth,
                    )} per month`
                  : risk.swr.withdrawal.type === 'default'
                  ? // Default should have been changed to asPercent if we are showing this.
                    assertFalse()
                  : noCase(risk.swr.withdrawal)}
              </h2>
            </>
          )
        }
        default:
          noCase(params.plan.advanced.strategy)
      }
    }
    case 'expected-returns': {
      const format = formatPercentage(1)
      const { stocks, bonds } =
        params.plan.advanced.annualReturns.expected.type === 'manual'
          ? params.plan.advanced.annualReturns.expected
          : EXPECTED_ANNUAL_RETURN_PRESETS(
              params.plan.advanced.annualReturns.expected.type,
              marketData.latest,
            )
      return (
        <>
          <h2>
            {expectedReturnTypeLabel(
              params.plan.advanced.annualReturns.expected,
            )}
          </h2>
          <h2>Stocks: {format(stocks)}</h2>
          <h2>Bonds: {format(bonds)}</h2>
        </>
      )
    }
    case 'inflation': {
      const format = formatPercentage(1)
      return (
        <h2>
          {inflationTypeLabel(params.plan.advanced.annualInflation)}:{' '}
          {format(
            params.plan.advanced.annualInflation.type === 'suggested'
              ? SUGGESTED_ANNUAL_INFLATION(marketData.latest)
              : params.plan.advanced.annualInflation.value,
          )}
        </h2>
      )
    }
    case 'strategy': {
      return (
        <>
          {(['TPAW', 'SPAW', 'SWR'] as const).map((value) => (
            <PlanInputChoiceItemSummary
              key={value}
              value={value}
              selected={(x) => params.plan.advanced.strategy === x}
              label={(x) => x}
            />
          ))}
        </>
      )
    }
    case 'simulation': {
      return params.plan.advanced.sampling === 'monteCarlo' ? (
        <>
          <h2>Monte Carlo Sequence</h2>
          <h2>
            Block Size:{' '}
            {numMonthsStr(params.plan.advanced.monteCarloSampling.blockSize)}
          </h2>
        </>
      ) : (
        <h2>Historical Sequence</h2>
      )
    }
    case 'dev':
      return (
        <div className="grid gap-y-1">
          {params.plan.advanced.annualReturns.historical.type !==
            'adjusted' && (
            <>
              <h2 className="font-medium">Historical Returns</h2>
              {(() => {
                switch (params.plan.advanced.annualReturns.historical.type) {
                  case 'unadjusted':
                    return <h2 className="ml-4">Unadjusted</h2>
                  case 'fixed':
                    return (
                      <>
                        <h2 className="ml-4">Fixed</h2>
                        <h2 className="ml-8">
                          Stocks:{' '}
                          {formatPercentage(1)(
                            params.plan.advanced.annualReturns.historical
                              .stocks,
                          )}
                        </h2>
                        <h2 className="ml-8">
                          Bonds:{' '}
                          {formatPercentage(1)(
                            params.plan.advanced.annualReturns.historical.bonds,
                          )}
                        </h2>
                      </>
                    )
                  default:
                    noCase(params.plan.advanced.annualReturns.historical)
                }
              })()}
            </>
          )}
          <h2 className="font-semibold">Misc</h2>
          {params.nonPlan.dev.alwaysShowAllMonths && (
            <h2 className="ml-4">Always Show All Months</h2>
          )}
          {params.plan.advanced.monteCarloSampling.numOfSimulations !==
            defaultParams.plan.advanced.monteCarloSampling.numOfSimulations && (
            <h2 className="ml-4">
              Num Simulations:{' '}
              {params.plan.advanced.monteCarloSampling.numOfSimulations}
            </h2>
          )}
          {params.nonPlan.dev.currentTimeFastForward.shouldFastForward && (
            <div>
              Current Time Offset:{' '}
              <h2 className="ml-4">
                years: {params.nonPlan.dev.currentTimeFastForward.years}
              </h2>
              <h2 className="ml-4">
                months: {params.nonPlan.dev.currentTimeFastForward.months}
              </h2>
              <h2 className="ml-4">
                days: {params.nonPlan.dev.currentTimeFastForward.days}
              </h2>
              <h2 className="ml-4">
                hours: {params.nonPlan.dev.currentTimeFastForward.hours}
              </h2>
            </div>
          )}
          {params.plan.risk.tpaw.additionalAnnualSpendingTilt !== 0 && (
            <>
              <h2 className="font-medium">Additional Spending Tilt</h2>
              <h2 className="ml-4">
                {formatPercentage(1)(
                  params.plan.risk.tpaw.additionalAnnualSpendingTilt,
                )}
              </h2>
            </>
          )}
        </div>
      )
    default:
      noCase(type)
  }
})

export const PlanInputValueForMonthRangeSummary = React.memo(
  ({
    entries,
    range,
  }: {
    entries: ValueForMonthRange[]
    range: SimpleRange
  }) => {
    if (entries.length === 0) return <h2>None</h2>
    return (
      <ol className={`list-outside list-disc ml-3 grid gap-y-2 mt-1`}>
        {entries.map((x, i) => (
          <li key={i} className=" break-inside-avoid-page">
            <ValueForMonthRangeDisplay
              entry={x}
              rangeAsMFN={range}
              skipLength={false}
            />
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
    intermediate: ReturnType<ParamsExtended['glidePathIntermediateValidated']>
    format: (x: number) => string
  }) => {
    const { paramsExt } = useSimulation()
    return (
      <>
        {intermediate.map((x, i) => (
          <React.Fragment key={i}>
            <h2 className={`${x.issue === 'none' ? '' : 'text-errorFG'}`}>
              {monthToStringForGlidePath(x.month, paramsExt).full}
            </h2>
            <h2 className="text-right">{format(x.stocks)}</h2>
          </React.Fragment>
        ))}
      </>
    )
  },
)

export const PlanInputGlidePathSummary = React.memo(
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
    const { intermediate, starting } = useMemo(
      () => ({
        intermediate: paramsExt.glidePathIntermediateValidated(
          glidePath.intermediate,
        ),
        starting: normalizeGlidePath(glidePath, paramsExt)[0],
      }),
      [glidePath, paramsExt],
    )

    return (
      <div
        className={`${className} inline-grid gap-x-10 items-center`}
        style={{ grid: 'auto/auto auto' }}
      >
        <h2>Now</h2>
        <h2 className="text-right">{format(starting)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter(
            (x) => x.issue !== 'before' && x.issue !== 'after',
          )}
          format={format}
        />
        <h2>At max age</h2>
        <h2 className="text-right">{format(glidePath.end.stocks)}</h2>
        <_GlidePathIntermediate
          intermediate={intermediate.filter((x) => x.issue === 'after')}
          format={format}
        />
      </div>
    )
  },
)

export function PlanInputChoiceItemSummary<Value>({
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
