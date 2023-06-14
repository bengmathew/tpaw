import {
  assertFalse,
  EXPECTED_ANNUAL_RETURN_PRESETS,
  fGet,
  noCase,
  RISK_TOLERANCE_VALUES,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { ReactNode } from 'react'
import { calendarMonthStr } from '../../Utils/CalendarMonthStr'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { numMonthsStr } from '../../Utils/NumMonthsStr'
import { trimAndNullify } from '../../Utils/TrimAndNullify'
import { yourOrYourPartners } from '../../Utils/YourOrYourPartners'
import { useMarketData } from '../App/WithMarketData'
import { useSimulation } from '../App/WithSimulation'
import { PlanInputCurrentPortfolioBalanceSummary } from '../Plan/PlanInput/PlanInputCurrentPortfolioBalance'
import { expectedReturnTypeLabel } from '../Plan/PlanInput/PlanInputExpectedReturns'
import { inflationTypeLabel } from '../Plan/PlanInput/PlanInputInflation'
import { isAdvancedInputModified } from '../Plan/PlanSummary/PlanSummary'
import {
  PlanInputChoiceItemSummary,
  PlanInputGlidePathSummary,
  PlanInputValueForMonthRangeSummary,
} from '../Plan/PlanSummary/PlanSummaryButton'
import { PrintSection } from './PrintSection'

// Note: The show/hide should mirror PlanSummary. "None" displays should mirror
// PlanSummaryButtons's empty attribute from PlanSummary.
export const PrintInputSection = React.memo(() => {
  const { params, paramsExt, defaultParams } = useSimulation()
  const { validMonthRangeAsMFN, allowFutureSavingsEntries } = paramsExt
  const isAdvancedModified = {
    'expected-returns': isAdvancedInputModified(
      'expected-returns',
      params,
      defaultParams,
    ),
    inflation: isAdvancedInputModified('inflation', params, defaultParams),
    simulation: isAdvancedInputModified('simulation', params, defaultParams),
    strategy: isAdvancedInputModified('strategy', params, defaultParams),
  }
  return (
    <>
      <PrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Your Inputs</h1>
      </PrintSection>
      <PrintSection
        className="grid gap-y-12 gap-x-5"
        style={{ grid: 'auto / auto 1fr' }}
      >
        {/* <h1 className="font-bold text-4xl text-center"> Your Inputs</h1> */}
        <_SectionHeading className="">Age</_SectionHeading>
        <div className="mt-1.5">
          <PlanInputAgeSummary />
        </div>
        <_SectionHeading className="">Wealth</_SectionHeading>
        <div className="mt-1.5">
          <div className=" break-inside-avoid-page">
            <_SubSectionHeading className="" isFirst>
              Current Portfolio Balance
            </_SubSectionHeading>
            <div className="mt-2">
              <PlanInputCurrentPortfolioBalanceSummary />
            </div>
          </div>
          {(allowFutureSavingsEntries ||
            params.plan.wealth.futureSavings.length > 0) && (
            <div className=" ">
              <_SubSectionHeading className="">
                Future Savings
              </_SubSectionHeading>
              <div className="mt-2">
                {params.plan.wealth.futureSavings.length > 0 ? (
                  <PlanInputValueForMonthRangeSummary
                    entries={params.plan.wealth.futureSavings}
                    range={validMonthRangeAsMFN('future-savings')}
                  />
                ) : (
                  <_None />
                )}
              </div>
            </div>
          )}
          <div className=" ">
            <_SubSectionHeading className="">
              Income During Retirement
            </_SubSectionHeading>
            <div className="mt-2">
              {params.plan.wealth.retirementIncome.length > 0 ? (
                <PlanInputValueForMonthRangeSummary
                  entries={params.plan.wealth.retirementIncome}
                  range={validMonthRangeAsMFN('income-during-retirement')}
                />
              ) : (
                <_None />
              )}
            </div>
          </div>
        </div>

        <_SectionHeading className="">
          Adjustments <br /> to <br /> Spending
        </_SectionHeading>

        <div className="mt-1.5">
          <_SubSectionHeading className="" isFirst>
            Extra Spending
          </_SubSectionHeading>
          <PlanInputExtraSpendingSummary forPrint />

          {params.plan.advanced.strategy !== 'SWR' && (
            <>
              <_SubSectionHeading className="">Legacy</_SubSectionHeading>
              <div className="mt-2">
                {params.plan.adjustmentsToSpending.tpawAndSPAW.legacy.external
                  .length === 0 &&
                params.plan.adjustmentsToSpending.tpawAndSPAW.legacy.total ===
                  0 ? (
                  <_None />
                ) : (
                  <PlanInputLegacySummary />
                )}
              </div>
            </>
          )}

          {params.plan.advanced.strategy !== 'SWR' && (
            <>
              <_SubSectionHeading className="">
                Spending Ceiling And Floor
              </_SubSectionHeading>
              <div className="mt-2">
                {params.plan.adjustmentsToSpending.tpawAndSPAW
                  .monthlySpendingCeiling === null &&
                params.plan.adjustmentsToSpending.tpawAndSPAW
                  .monthlySpendingFloor === null ? (
                  <_None />
                ) : (
                  <PlanInputSpendingCeilingAndFloorSummary />
                )}
              </div>
            </>
          )}
        </div>
        <_SectionHeading className="">Risk</_SectionHeading>
        <div className=" break-inside-avoid-page mt-2">
          <div className="">
            <PlanInputRiskSummary />
          </div>
        </div>

        <_SectionHeading className="">Advanced</_SectionHeading>
        <div className="mt-1.5">
          <_SubSectionHeading className="" isFirst>
            Expected Returns
          </_SubSectionHeading>
          <div className="mt-2">
            <PlanInputExpectedReturnsSummary />
          </div>
          <_SubSectionHeading className="">Inflation</_SubSectionHeading>
          <div className="mt-2">
            <PlanInputInflationSummary />
          </div>
          <_SubSectionHeading className="">Simulation</_SubSectionHeading>
          <div className="mt-2">
            <PlanInputSimulationSummary />
          </div>
          <_SubSectionHeading className="">Strategy</_SubSectionHeading>
          <div className="mt-2">
            <PlanInputStrategySummary />
          </div>
        </div>
      </PrintSection>
    </>
  )
})

const _SectionHeading = React.memo(
  ({ className, children }: { className?: string; children: ReactNode }) => {
    return (
      <h2
        className={clsx(
          className,
          'font-bold text-3xl text-right border-r-2 border-black px-5',
        )}
      >
        {children}
      </h2>
    )
  },
)
const _SubSectionHeading = React.memo(
  ({
    className,
    children,
    isFirst = false,
  }: {
    className?: string
    children: ReactNode
    isFirst?: boolean
  }) => {
    return (
      <h2 className={clsx(className, 'font-bold text-xl', !isFirst && 'mt-6')}>
        {children}
      </h2>
    )
  },
)

const _None = React.memo(() => (
  <div className="">
    <h2 className="">None</h2>
  </div>
))

export const PlanInputAgeSummary = React.memo(() => {
  const { params, paramsExt } = useSimulation()
  const {
    isAgesNotRetired,
    isPersonRetired,
    pickPerson,
    getCurrentAgeOfPerson,
  } = paramsExt
  if (params.plan.dialogPosition === 'age') {
    return (
      <>
        <h2>Month of Birth: </h2>
        <h2>Retirement: </h2>
        <h2>Max: </h2>
      </>
    )
  }
  const forPerson = (person: 'person1' | 'person2', className = '') => {
    const { ages } = pickPerson(person)
    return isAgesNotRetired(ages) ? (
      <>
        <h2 className={`${className}`}>Month of Birth</h2>
        <h2>
          {calendarMonthStr(ages.monthOfBirth)} (Age:{' '}
          {numMonthsStr(getCurrentAgeOfPerson(person).inMonths)})
        </h2>
        <h2 className={`${className}`}>Retirement</h2>
        <h2> {numMonthsStr(ages.retirementAge.inMonths)}</h2>
        <h2 className={`${className}`}>Max</h2>
        <h2> {numMonthsStr(ages.maxAge.inMonths)}</h2>
      </>
    ) : (
      <>
        <h2 className={`${className} col-span-2`}>Retired</h2>
        <h2 className={`${className}`}>Month of Birth</h2>
        <h2>
          {' '}
          {calendarMonthStr(ages.monthOfBirth)} (Age:{' '}
          {numMonthsStr(getCurrentAgeOfPerson(person).inMonths)})
        </h2>
        <h2 className={`${className}`}>Max</h2>
        <h2> {numMonthsStr(ages.maxAge.inMonths)}</h2>
      </>
    )
  }
  if (params.plan.people.withPartner) {
    const withdrawalPerson = pickPerson(params.plan.people.withdrawalStart)
    return (
      <div
        className={clsx('grid gap-x-3 gap-y-1')}
        style={{ grid: 'auto/auto 1fr' }}
      >
        <h2 className="font-medium col-span-2">You</h2>
        {forPerson('person1', 'ml-4')}
        <h2 className="mt-2 font-medium  col-span-2">Your Partner</h2>
        {forPerson('person2', 'ml-4')}
        {!(isPersonRetired('person1') && isPersonRetired('person2')) && (
          <h2 className="mt-2  col-span-2">
            Withdrawals start{' '}
            {isPersonRetired(withdrawalPerson)
              ? 'now.'
              : `at ${yourOrYourPartners(
                  params.plan.people.withdrawalStart,
                )} retirement.`}
          </h2>
        )}
      </div>
    )
  } else {
    return (
      <div
        className={clsx('grid gap-x-3 gap-y-1')}
        style={{ grid: 'auto/auto 1fr' }}
      >
        {forPerson('person1')}
      </div>
    )
  }
})

export const PlanInputLegacySummary = React.memo(() => {
  const { params, paramsProcessed } = useSimulation()
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
})

export const PlanInputSpendingCeilingAndFloorSummary = React.memo(() => {
  const { params } = useSimulation()
  const { monthlySpendingCeiling, monthlySpendingFloor } =
    params.plan.adjustmentsToSpending.tpawAndSPAW
  return (
    <>
      <h2>
        Ceiling:{' '}
        {monthlySpendingCeiling
          ? `${formatCurrency(monthlySpendingCeiling)} per month`
          : 'None'}
      </h2>
      <h2>
        Floor:{' '}
        {monthlySpendingFloor
          ? `${formatCurrency(monthlySpendingFloor)} per month`
          : 'None'}
      </h2>
    </>
  )
})

export const PlanInputExtraSpendingSummary = React.memo(
  ({ forPrint = false }: { forPrint?: boolean }) => {
    const { params, paramsExt } = useSimulation()
    const { validMonthRangeAsMFN } = paramsExt
    const { essential, discretionary } =
      params.plan.adjustmentsToSpending.extraSpending
    const showLabels = params.plan.advanced.strategy !== 'SWR'
    return (
      <>
        {essential.length === 0 && discretionary.length === 0 && <h2>None</h2>}
        {essential.length > 0 && (
          <>
            {showLabels && (
              <h2
                className={clsx(
                  'mt-1 font-medium ',
                  forPrint && 'text-lg mt-2',
                )}
              >
                Essential
              </h2>
            )}
            <PlanInputValueForMonthRangeSummary
              entries={essential}
              range={validMonthRangeAsMFN('extra-spending')}
            />
          </>
        )}
        {discretionary.length > 0 && (
          <>
            {showLabels && (
              <h2
                className={clsx(
                  'mt-1 font-medium ',
                  forPrint && 'text-lg mt-2',
                )}
              >
                Discretionary
              </h2>
            )}
            <PlanInputValueForMonthRangeSummary
              entries={discretionary}
              range={validMonthRangeAsMFN('extra-spending')}
            />
          </>
        )}
      </>
    )
  },
)

export const PlanInputRiskSummary = React.memo(() => {
  const { params } = useSimulation()
  switch (params.plan.advanced.strategy) {
    case 'TPAW':
      return <PlanInputRiskTPAWSummary />
    case 'SPAW':
      return <PlanInputRiskSPAWSummary />
    case 'SWR':
      return <PlanInputRiskSWRSummary />
    default:
      noCase(params.plan.advanced.strategy)
  }
})

export const PlanInputRiskTPAWSummary = React.memo(() => {
  const { params, defaultParams } = useSimulation()
  const { risk } = params.plan
  const defaultRisk = defaultParams.plan.risk
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
          {defaultRisk.tpaw.timePreference !== risk.tpaw.timePreference && (
            <h2 className="ml-4">
              Spending Tilt: {formatPercentage(1)(-risk.tpaw.timePreference)}
            </h2>
          )}
        </>
      )}
    </>
  )
})

export const PlanInputRiskSPAWSummary = React.memo(() => {
  const { params } = useSimulation()
  const { risk } = params.plan
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
        Spending Tilt: {formatPercentage(1)(risk.spaw.annualSpendingTilt)}
      </h2>
    </>
  )
})

export const PlanInputRiskSWRSummary = React.memo(() => {
  const { params, paramsExt } = useSimulation()
  const { risk } = params.plan
  const { withdrawalsStarted } = paramsExt
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
          ? `${formatPercentage(1)(risk.swr.withdrawal.percentPerYear)} of ${
              withdrawalsStarted
                ? 'current portfolio balance'
                : 'savings portfolio at retirement'
            }`
          : risk.swr.withdrawal.type === 'asAmountPerMonth'
          ? `${formatCurrency(risk.swr.withdrawal.amountPerMonth)} per month`
          : risk.swr.withdrawal.type === 'default'
          ? // Default should have been changed to asPercent if we are showing this.
            assertFalse()
          : noCase(risk.swr.withdrawal)}
      </h2>
    </>
  )
})

export const PlanInputExpectedReturnsSummary = React.memo(() => {
  const { params } = useSimulation()
  const marketData = useMarketData()
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
        {expectedReturnTypeLabel(params.plan.advanced.annualReturns.expected)}
      </h2>
      <h2>Stocks: {format(stocks)}</h2>
      <h2>Bonds: {format(bonds)}</h2>
    </>
  )
})
export const PlanInputInflationSummary = React.memo(() => {
  const { params } = useSimulation()
  const marketData = useMarketData()
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
})
export const PlanInputSimulationSummary = React.memo(() => {
  const { params } = useSimulation()
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
})
export const PlanInputStrategySummary = React.memo(() => {
  const { params } = useSimulation()
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
})
