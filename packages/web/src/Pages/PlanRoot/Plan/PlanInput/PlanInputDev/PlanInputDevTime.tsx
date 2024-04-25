import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, block, fGet, noCase } from '@tpaw/common'
import clix from 'clsx'
import _, { capitalize } from 'lodash'
import { DateTime, Duration } from 'luxon'
import React from 'react'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { writableCloneDeep } from '../../../../../Utils/WritableCloneDeep'
import { NumberInput } from '../../../../Common/Inputs/NumberInput'
import { SwitchAsToggle } from '../../../../Common/Inputs/SwitchAsToggle'
import { useMarketData } from '../../../PlanRootHelpers/WithMarketData'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfo,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevFastForward = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <>
          <_FastForwardCard className="mt-10" props={props} />
          <_SynthesizeMarketDataCard className="mt-10" props={props} />
        </>
      </PlanInputBody>
    )
  },
)

const _FastForwardCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNorm, fastForwardInfo } = useSimulation()
    const { datingInfo } = planParamsNorm
    const currentTimestamp = datingInfo.isDated
      ? datingInfo.nowAsTimestamp
      : datingInfo.nowAsTimestampNominal
    const { getZonedTime } = useIANATimezoneName()
    const currentTime: DateTime = getZonedTime(currentTimestamp)
    const { portfolioBalance } = planParamsNorm.wealth
    const portfolioBalanceUpdatedAt = portfolioBalance.isDatedPlan
      ? portfolioBalance.updatedHere
        ? planParamsNorm.timestamp
        : portfolioBalance.updatedAtTimestamp
      : null

    const isModified = useIsFastForwardCardModified()

    const formatDistanceFromNow = (x: DateTime | number) => {
      const diff = getZonedTime(currentTimestamp).diff(
        DateTime.fromMillis(x.valueOf()),
        ['years', 'months', 'days', 'hours'],
      )
      return Duration.fromObject(
        _.mapValues(diff.toObject(), (x) => Math.round(x as number)),
      )
        .rescale()
        .toHuman()
    }

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="-mt-2 text-sm text-right">
          Zone: {currentTime.toFormat('ZZZZ')}
        </h2>
        <div
          className="grid mt-2 gap-x-4 gap-y-2"
          style={{ grid: 'auto/auto 1fr' }}
        >
          <h2 className="text-right">Current</h2>
          <h2 className="font-mono text-sm">{_formatDateTime(currentTime)}</h2>
          <h2 className="text-right">Params</h2>
          <div className="">
            <h2 className="font-mono text-sm">
              {_formatDateTime(getZonedTime(planParamsNorm.timestamp))}
            </h2>
            <h2 className="text-sm font-font1 lighten">
              {formatDistanceFromNow(planParamsNorm.timestamp)} ago
            </h2>
          </div>
          <h2 className="text-right">Portfolio</h2>
          <div className="">
            {portfolioBalanceUpdatedAt ? (
              <>
                <h2 className="font-mono text-sm">
                  {_formatDateTime(getZonedTime(portfolioBalanceUpdatedAt))}
                </h2>
                <h2 className="text-sm font-font1 lighten">
                  {formatDistanceFromNow(portfolioBalanceUpdatedAt)} ago
                </h2>
              </>
            ) : (
              <h2 className="font-mono text-sm">N/A (undated plan)</h2>
            )}
          </div>
        </div>
        <div className=" flex justify-start gap-x-4 items-center mt-8">
          <h2 className={clix(`font-semibold`)}>Fast Forward</h2>
          <SwitchAsToggle
            className="disabled:lighten-2"
            checked={fastForwardInfo.isFastForwarding}
            setChecked={() => {
              if (!fastForwardInfo.isFastForwarding) {
                fastForwardInfo.setFastForwardSpec({
                  years: 0,
                  months: 0,
                  days: 0,
                  hours: 0,
                })
              } else {
                // No way to disable fast forward.
              }
            }}
          />
        </div>
        {fastForwardInfo.isFastForwarding ? (
          <div className="">
            <div
              className="items-center grid gap-x-2 gap-y-2 mt-4 "
              style={{ grid: 'auto/auto 1fr' }}
            >
              <_FastForwardInput
                type="years"
                fastForwardInfo={fastForwardInfo}
              />
              <_FastForwardInput
                type="months"
                fastForwardInfo={fastForwardInfo}
              />
              <_FastForwardInput
                type="days"
                fastForwardInfo={fastForwardInfo}
              />
              <_FastForwardInput
                type="hours"
                fastForwardInfo={fastForwardInfo}
              />
            </div>
          </div>
        ) : (
          <div>
            Note: Make a copy of this plan before fast forwarding. Fast forward
            cannot be undone.
          </div>
        )}
      </div>
    )
  },
)

const _FastForwardInput = React.memo(
  ({
    type,
    fastForwardInfo,
  }: {
    type: 'years' | 'months' | 'days' | 'hours'
    fastForwardInfo: Extract<
      SimulationInfo['fastForwardInfo'],
      { isFastForwarding: true }
    >
  }) => {
    const { spec, setFastForwardSpec } = fastForwardInfo
    return (
      <>
        <h2 className="">{capitalize(type)}</h2>
        <NumberInput
          className=""
          showDecrement={false}
          value={spec[type]}
          setValue={(x: number) => {
            const clone = _.cloneDeep(spec)
            assert(clone)
            const clamped = Math.max(x, clone[type])
            clone[type] = clamped
            setFastForwardSpec(clone)
            return clamped !== x
          }}
          buttonClassName="px-2 py-1"
          // width?: number
          modalLabel={'Years'}
        />
      </>
    )
  },
)

const _SynthesizeMarketDataCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const isModified = useIsSynthesizeMarketDataCardModified()
    const { planParamsNorm, simulationResult } = useSimulation()
    const {args} = simulationResult
    const {
      synthesizeMarketDataSpec,
      setSynthesizeMarketDataSpec,
      applySynthesizeMarketDataSpec,
    } = useMarketData()

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <div className="flex items-center gap-x-4 mb-2">
          <h2 className="font-bold text-lg ">Synthesize Market Data</h2>

          <SwitchAsToggle
            className=""
            checked={synthesizeMarketDataSpec !== null}
            setChecked={(enabled) => {
              setSynthesizeMarketDataSpec(
                enabled
                  ? {
                      v: 1,
                      yearsBeforeNow: 0,
                      yearsAfterNow: 0,
                      strategy: {
                        dailyStockMarketPerformance: {
                          type: 'roundRobinOfRealData',
                        },
                      },
                    }
                  : null,
              )
            }}
          />
        </div>
        {synthesizeMarketDataSpec && (
          <div className="mt-4">
            <div
              className="grid gap-x-4 gap-y-2 mt-6"
              style={{ grid: 'auto/auto 1fr' }}
            >
              <h2 className="text-right">Years Before Now</h2>
              <NumberInput
                className=""
                value={synthesizeMarketDataSpec.yearsBeforeNow}
                setValue={(value: number) => {
                  assert(synthesizeMarketDataSpec)
                  const clone = writableCloneDeep(synthesizeMarketDataSpec)
                  const clamped = Math.max(0, value)
                  if (clamped === clone.yearsBeforeNow) return true
                  clone.yearsBeforeNow = clamped
                  setSynthesizeMarketDataSpec(clone)
                  return false
                }}
                modalLabel="Years Before Now"
              />
              <h2 className="text-right">Years After Now</h2>
              <NumberInput
                className=""
                value={synthesizeMarketDataSpec.yearsAfterNow}
                setValue={(value: number) => {
                  assert(synthesizeMarketDataSpec)
                  const clone = writableCloneDeep(synthesizeMarketDataSpec)
                  const clamped = Math.max(0, value)
                  if (clamped === clone.yearsAfterNow) return true
                  clone.yearsAfterNow = clamped
                  setSynthesizeMarketDataSpec(clone)
                  return false
                }}
                modalLabel="Years After Now"
              />
            </div>
            <h2 className="font-semibold mt-6">
              CAPE, Bond Rates, and Inflation:{' '}
            </h2>
            <div className="flex items-start gap-x-2 cursor-pointer mt-2">
              <FontAwesomeIcon
                className="text-sm mt-1.5"
                icon={faCircleSelected}
              />
              <h2 className="">
                Round robin of last 30 days before fast forward.
              </h2>
            </div>
            <h2 className="font-semibold mt-6 mb-2">
              Daily VT and BND Performance{' '}
            </h2>

            <div
              className=""
              onClick={() => {
                assert(synthesizeMarketDataSpec)
                setSynthesizeMarketDataSpec({
                  ...synthesizeMarketDataSpec,
                  strategy: {
                    dailyStockMarketPerformance: {
                      type: 'constant',
                      annualBND:
                        args.planParamsProcessed.returnsStatsForPlanning.stocks
                          .empiricalAnnualNonLogExpectedReturnInfo.value,
                      annualVT:
                        args.planParamsProcessed.returnsStatsForPlanning.bonds
                          .empiricalAnnualNonLogExpectedReturnInfo.value,
                    },
                  },
                })
              }}
            >
              <div className="flex items-start gap-x-2 cursor-pointer">
                <FontAwesomeIcon
                  className="text-sm mt-1.5"
                  icon={
                    synthesizeMarketDataSpec.strategy
                      .dailyStockMarketPerformance.type === 'constant'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />
                <div className="">
                  <h2 className="">
                    Constant (copy from expected annual return)
                  </h2>
                  {synthesizeMarketDataSpec.strategy.dailyStockMarketPerformance
                    .type === 'constant' &&
                    block(() => {
                      const strategy = fGet(synthesizeMarketDataSpec).strategy
                        .dailyStockMarketPerformance
                      assert(strategy.type === 'constant')

                      return (
                        <>
                          <div
                            className="grid gap-x-4 gap-y-2 mt-2"
                            style={{ grid: 'auto/auto 1fr' }}
                          >
                            <h2 className="text-right">VT Annual Return:</h2>
                            <h2 className="">
                              {formatPercentage(1)(strategy.annualVT)}
                            </h2>
                            <h2 className="text-right">BND Annual Return:</h2>
                            <h2 className="">
                              {formatPercentage(1)(strategy.annualBND)}
                            </h2>
                          </div>
                          <div className="mt-2">
                            {planParamsNorm.advanced.returnsStatsForPlanning
                              .expectedValue.empiricalAnnualNonLog.type ===
                            'fixed' ? (
                              <h2 className="">
                                NOTE: Expected return is fixed.
                              </h2>
                            ) : (
                              <h2 className="text-errorFG">
                                NOTE: Expected return is NOT fixed.
                              </h2>
                            )}
                          </div>
                        </>
                      )
                    })}
                </div>
              </div>
            </div>
            <button
              className="block mt-1.5"
              onClick={() => {
                assert(synthesizeMarketDataSpec)
                setSynthesizeMarketDataSpec({
                  ...synthesizeMarketDataSpec,
                  strategy: {
                    dailyStockMarketPerformance: {
                      type: 'roundRobinOfRealData',
                    },
                  },
                })
              }}
            >
              <div className="flex items-start gap-x-2 cursor-pointer">
                <FontAwesomeIcon
                  className="text-sm mt-1.5"
                  icon={
                    synthesizeMarketDataSpec.strategy
                      .dailyStockMarketPerformance.type ===
                    'roundRobinOfRealData'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />
                <h2 className="">Round robin of real data.</h2>
              </div>
            </button>
            <button
              className="block mt-1.5"
              onClick={() => {
                assert(synthesizeMarketDataSpec)
                setSynthesizeMarketDataSpec({
                  ...synthesizeMarketDataSpec,
                  strategy: {
                    dailyStockMarketPerformance: {
                      type: 'repeatGrowShrinkZero',
                    },
                  },
                })
              }}
            >
              <div className="flex items-start gap-x-2 cursor-pointer">
                <FontAwesomeIcon
                  className="text-sm mt-1.5"
                  icon={
                    synthesizeMarketDataSpec.strategy
                      .dailyStockMarketPerformance.type ===
                    'repeatGrowShrinkZero'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />
                <h2 className="">Cycle through grow (5%), shrink, flat.</h2>
              </div>
            </button>
          </div>
        )}
        <div className="flex justify-end mt-4">
          <button
            className="btn-sm btn-dark disabled:lighten-2"
            disabled={!applySynthesizeMarketDataSpec}
            onClick={() => fGet(applySynthesizeMarketDataSpec)()}
          >
            Apply
          </button>
        </div>
      </div>
    )
  },
)

export const useIsPlanInputDevTimeModified = () => {
  const m1 = useIsFastForwardCardModified()
  const m2 = useIsSynthesizeMarketDataCardModified()
  return m1 || m2
}

const useIsFastForwardCardModified = () => {
  const { fastForwardInfo } = useSimulation()
  return fastForwardInfo.isFastForwarding
}

const useIsSynthesizeMarketDataCardModified = () => {
  const { synthesizeMarketDataSpec } = useMarketData()
  return synthesizeMarketDataSpec !== null
}

export const PlanInputDevTimeSummary = React.memo(() => {
  const { synthesizeMarketDataSpec } = useMarketData()
  const { getZonedTime } = useIANATimezoneName()
  const { fastForwardInfo, planParamsNorm } = useSimulation()
  const { datingInfo } = planParamsNorm
  const currentTimestamp = datingInfo.isDated
    ? datingInfo.nowAsTimestamp
    : datingInfo.nowAsTimestampNominal

  return (
    <>
      <h2 className="">
        Current: {_formatDateTime(getZonedTime(currentTimestamp))}
      </h2>
      <h2>Fast Forward:</h2>
      {fastForwardInfo.isFastForwarding ? (
        <div>
          <h2 className="ml-4">years: {fastForwardInfo.spec.years}</h2>
          <h2 className="ml-4">months: {fastForwardInfo.spec.months}</h2>
          <h2 className="ml-4">days: {fastForwardInfo.spec.days}</h2>
          <h2 className="ml-4">hours: {fastForwardInfo.spec.hours}</h2>
        </div>
      ) : (
        <h2 className="ml-4">None</h2>
      )}
      <h2>Synthesize Market Data:</h2>
      {synthesizeMarketDataSpec ? (
        <div className="ml-4">
          <h2 className="">
            Years Before Now: {synthesizeMarketDataSpec.yearsBeforeNow}
          </h2>
          <h2 className="">
            Years After Now: {synthesizeMarketDataSpec.yearsAfterNow}
          </h2>
          <h2 className="">
            {' '}
            Strategy For Daily Stock Market Performance:{' '}
            {block(() => {
              const { dailyStockMarketPerformance } =
                synthesizeMarketDataSpec.strategy
              return (
                <div className="ml-4">
                  <h2 className="">{dailyStockMarketPerformance.type}</h2>
                  {dailyStockMarketPerformance.type === 'constant' ? (
                    <>
                      <h2 className="">
                        Annual VT:
                        {formatPercentage(1)(
                          dailyStockMarketPerformance.annualVT,
                        )}
                      </h2>
                      <h2 className="">
                        Annual BND:
                        {formatPercentage(1)(
                          dailyStockMarketPerformance.annualBND,
                        )}
                      </h2>
                    </>
                  ) : dailyStockMarketPerformance.type ===
                    'repeatGrowShrinkZero' ? (
                    <></>
                  ) : dailyStockMarketPerformance.type ===
                    'roundRobinOfRealData' ? (
                    <></>
                  ) : (
                    noCase(dailyStockMarketPerformance)
                  )}
                </div>
              )
            })}
          </h2>
        </div>
      ) : (
        <h2 className="ml-4">None</h2>
      )}
    </>
  )
})

const _formatDateTime = (x: DateTime) => x.toFormat('LL/d/yyyy - HH:mm EEE')
