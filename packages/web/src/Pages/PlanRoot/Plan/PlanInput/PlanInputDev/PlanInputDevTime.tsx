import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert } from '@tpaw/common'
import clix from 'clsx'
import _, { capitalize } from 'lodash'
import { DateTime, Duration } from 'luxon'
import React from 'react'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { NumberInput } from '../../../../Common/Inputs/NumberInput'
import { SwitchAsToggle } from '../../../../Common/Inputs/SwitchAsToggle'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfo,
  useDailyMarketSeriesSrc,
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {
  useIsPlanInputDevTimeFastForwardCardModified,
  useIsPlanInputDevTimeSynthesizeMarketDataCardModified,
} from './PlanInputDevTimeFns'

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
    const { planParamsNormInstant, fastForwardInfo } = useSimulationInfo()
    const { datingInfo } = planParamsNormInstant
    const currentTimestamp = datingInfo.isDated
      ? datingInfo.nowAsTimestamp
      : datingInfo.nowAsTimestampNominal
    const { getZonedTime } = useIANATimezoneName()
    const currentTime: DateTime = getZonedTime(currentTimestamp)
    const { portfolioBalance } = planParamsNormInstant.wealth
    const portfolioBalanceUpdatedAt = portfolioBalance.isDatedPlan
      ? portfolioBalance.updatedHere
        ? planParamsNormInstant.timestamp
        : portfolioBalance.updatedAtTimestamp
      : null

    const isModified = useIsPlanInputDevTimeFastForwardCardModified()

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
              {_formatDateTime(getZonedTime(planParamsNormInstant.timestamp))}
            </h2>
            <h2 className="text-sm font-font1 lighten">
              {formatDistanceFromNow(planParamsNormInstant.timestamp)} ago
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
    const isModified = useIsPlanInputDevTimeSynthesizeMarketDataCardModified()
    const { planParamsProcessed } = useSimulationResultInfo().simulationResult
    const { dailyMarketSeriesSrc, setDailyMarketSeriesSrc } =
      useDailyMarketSeriesSrc()

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <h2 className="font-bold text-lg ">Daily Market Series Source</h2>
        <div className="mt-2">
          <button
            className="flex items-center gap-x-2 py-2"
            onClick={() => setDailyMarketSeriesSrc({ type: 'live' })}
          >
            <FontAwesomeIcon
              icon={
                dailyMarketSeriesSrc.type === 'live'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />
            Live
          </button>
          <button
            className="flex items-start  gap-x-2 py-2"
            onClick={() => {
              setDailyMarketSeriesSrc({
                type: 'syntheticConstant',
                annualPercentageChangeVT:
                  planParamsProcessed.returnsStatsForPlanning.stocks
                    .empiricalAnnualNonLogExpectedReturn,
                annualPercentageChangeBND:
                  planParamsProcessed.returnsStatsForPlanning.bonds
                    .empiricalAnnualNonLogExpectedReturn,
              })
            }}
          >
            <FontAwesomeIcon
              className="mt-1"
              icon={
                dailyMarketSeriesSrc.type === 'syntheticConstant'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />

            <div className="text-start">
              <div className="">Synthetic Constant</div>

              {dailyMarketSeriesSrc.type === 'syntheticConstant' && (
                <>
                  <div className="">
                    VT:{' '}
                    {formatPercentage(1)(
                      dailyMarketSeriesSrc.annualPercentageChangeVT,
                    )}
                  </div>
                  <div className="">
                    BND:{' '}
                    {formatPercentage(1)(
                      dailyMarketSeriesSrc.annualPercentageChangeBND,
                    )}
                  </div>
                  <div className="text-sm lighten">
                    VT and BND are copied from expected annual return.
                  </div>
                  <div className="text-sm lighten">
                    The other market data series like Inflation, SP500, and Bond
                    Rates are fixed to some constant. Check the code for the
                    constants
                  </div>
                </>
              )}
            </div>
          </button>
          <button
            className="flex items-center gap-x-2 py-2"
            onClick={() =>
              setDailyMarketSeriesSrc({ type: 'syntheticLiveRepeated' })
            }
          >
            <FontAwesomeIcon
              icon={
                dailyMarketSeriesSrc.type === 'syntheticLiveRepeated'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />
            Synthetic Live Repeated
          </button>
        </div>
      </div>
    )
  },
)

export const PlanInputDevTimeSummary = React.memo(() => {
  const { dailyMarketSeriesSrc } = useDailyMarketSeriesSrc()
  const { getZonedTime } = useIANATimezoneName()
  const { fastForwardInfo, planParamsNormInstant } = useSimulationInfo()
  const { datingInfo } = planParamsNormInstant
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
      <h2>Market Data Series Source:</h2>
      {dailyMarketSeriesSrc.type === 'live' ? (
        <h2 className="ml-4">Live</h2>
      ) : dailyMarketSeriesSrc.type === 'syntheticConstant' ? (
        <>
          <h2 className="ml-4">
            {`Synthetic Constant (VT: ${formatPercentage(1)(
              dailyMarketSeriesSrc.annualPercentageChangeVT,
            )} BND: ${formatPercentage(1)(
              dailyMarketSeriesSrc.annualPercentageChangeBND,
            )})`}
          </h2>
        </>
      ) : (
        <h2 className="ml-4">Synthetic Live Repeated</h2>
      )}
    </>
  )
})

const _formatDateTime = (x: DateTime) => x.toFormat('LL/d/yyyy - HH:mm EEE')
