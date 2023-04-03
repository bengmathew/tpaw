import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'

import {
  faCircle as faCircleSelected,
  faMinus,
  faPlus,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES,
  Params,
  PlanParams,
  assert,
  getDefaultPlanParams,
  planParamsGuard,
} from '@tpaw/common'
import _, { capitalize } from 'lodash'
import { DateTime, Duration } from 'luxon'
import Link from 'next/link'
import React, { useEffect, useMemo, useState } from 'react'
// import {
//   generateSampledAnnualReturnStatsTable,
//   getAnnualToMonthlyRateConvertionCorrection,
// } from '../../../TPAWSimulator/PlanParamsProcessed/GetAnnualToMonthlyRateConvertionCorrection'
import { RadioGroup } from '@headlessui/react'
import { clearMemoizedRandom } from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { NumberInput } from '../../Common/Inputs/NumberInput'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { ToggleSwitch } from '../../Common/Inputs/ToggleSwitch'
import { useGetPlanChartURL } from '../PlanChart/UseGetPlanChartURL'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputDev = React.memo((props: PlanInputBodyPassThruProps) => {
  return (
    <PlanInputBody {...props}>
      <div className="">
        <_HistoricalCReturnsCard className="" props={props} />
        <_MiscCard className="mt-10" props={props} />
        <_TimeCard className="mt-10" props={props} />
        <_AdditionalSpendingTiltCard className="mt-10" props={props} />
      </div>
    </PlanInputBody>
  )
})

const _HistoricalCReturnsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setPlanParams, defaultParams, paramsProcessed } =
      useSimulation()

    const defaultHistorical =
      defaultParams.plan.advanced.annualReturns.historical
    const isModified = !_.isEqual(
      defaultHistorical,
      params.plan.advanced.annualReturns.historical,
    )

    const handleChange = (
      historical: PlanParams['advanced']['annualReturns']['historical'],
    ) => {
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        clone.advanced.annualReturns.historical = historical
        return clone
      })
    }
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg">Historical Returns</h2>

        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'adjusted',
                adjustment: { type: 'toExpected' },
                correctForBlockSampling: true,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                params.plan.advanced.annualReturns.historical.type ===
                'adjusted'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjusted to Expected
          </h2>
          {params.plan.advanced.annualReturns.historical.type ===
            'adjusted' && (
            <_HistoricalAdjustedDetails
              className="ml-[28px] mt-2"
              historical={params.plan.advanced.annualReturns.historical}
            />
          )}
        </div>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() => handleChange({ type: 'unadjusted' })}
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                params.plan.advanced.annualReturns.historical.type ===
                'unadjusted'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Unadjusted
          </h2>
        </div>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'fixed',
                stocks: paramsProcessed.returns.expectedAnnualReturns.stocks,
                bonds: paramsProcessed.returns.expectedAnnualReturns.bonds,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                params.plan.advanced.annualReturns.historical.type === 'fixed'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Fixed
          </h2>
          {params.plan.advanced.annualReturns.historical.type === 'fixed' && (
            <_HistoricalFixedDetails className="ml-[28px] mt-2" />
          )}
        </div>

        <h2 className="font-semibold mt-6">
          Adjustment Corrections for Block Sampling
        </h2>

        <div className="ml-4 flex gap-x-4">
          <button
            className="mt-4 underline"
            onClick={async () => {
              // console.dir(await generateSampledAnnualReturnStatsTable())
            }}
          >
            Generate
          </button>
          <button
            className="mt-4 underline block"
            onClick={() => {
              // console.dir(
              //   sampledReturnsStatsTableRaw.map((x) => [
              //     x.blockSize,
              //     x.stocks.oneYear.mean,
              //     x.stocks.oneYear.ofLog.varianceAveragedOverThread,
              //     x.bonds.oneYear.mean,
              //   ]),
              // )
            }}
          >
            Minify
          </button>

          <button
            className="block pt-4 underline"
            onClick={() => {
              // const windowSizes = ['one', 'five', 'ten', 'thirty'] as const
              // const titles = (windowSize: typeof windowSizes[number]) =>
              //   [
              //     'Stocks Expected Value',
              //     'Stocks Expected Value of Log',
              //     'Stocks Standard Deviation of Log',
              //     'Bonds Expected Value',
              //     'Bonds Expected Value of Log',
              //     'Bonds Standard Deviation of Log',
              //   ].map((x) => `${windowSize.toUpperCase()} Year ${x}`)
              // const values = (
              //   windowSize: typeof windowSizes[number],
              //   { stocks, bonds }: typeof sampledReturnsStatsTableRaw[number],
              // ) => [
              //   `${stocks[`${windowSize}Year`].mean}`,
              //   `${stocks[`${windowSize}Year`].ofLog.mean}`,
              //   `${Math.sqrt(
              //     stocks[`${windowSize}Year`].ofLog.varianceAveragedOverThread,
              //   )}`,
              //   `${bonds[`${windowSize}Year`].mean}`,
              //   `${bonds[`${windowSize}Year`].ofLog.mean}`,
              //   `${Math.sqrt(
              //     bonds[`${windowSize}Year`].ofLog.varianceAveragedOverThread,
              //   )}`,
              // ]
              // const csv = [
              //   ['Block Size', ..._.flatten(windowSizes.map(titles))].join(','),
              //   ...sampledReturnsStatsTableRaw.map((row) =>
              //     [
              //       `${row.blockSize}`,
              //       ..._.flatten(
              //         windowSizes.map((windowSize) => values(windowSize, row)),
              //       ),
              //     ].join(','),
              //   ),
              // ].join('\n')
              // void navigator.clipboard.writeText(csv)
            }}
          >
            Copy as CSV
          </button>
        </div>
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() => handleChange(defaultHistorical)}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _HistoricalAdjustedDetails = React.memo(
  ({
    className = '',
    historical,
  }: {
    className?: string
    historical: Extract<
      PlanParams['advanced']['annualReturns']['historical'],
      { type: 'adjusted' }
    >
  }) => {
    const { setPlanParams } = useSimulation()
    const { adjustment, correctForBlockSampling } = historical
    assert(adjustment.type === 'toExpected')
    return (
      <div className={`${className} ml-8 `}>
        <div className="flex  items-center gap-x-4  py-1.5">
          <h2 className="">Correct for Block Sampling</h2>
          <ToggleSwitch
            className=""
            enabled={correctForBlockSampling}
            setEnabled={(x) =>
              setPlanParams((plan) => {
                const clone = _.cloneDeep(plan)
                const historicalClone = _.cloneDeep(historical)
                historicalClone.correctForBlockSampling = x
                clone.advanced.annualReturns.historical = historicalClone
                return clone
              })
            }
          />
        </div>
      </div>
    )
  },
)
const _HistoricalFixedDetails = React.memo(
  ({ className = '' }: { className?: string }) => {
    const _delta = 0.1
    const { params, setPlanParams } = useSimulation()
    assert(params.plan.advanced.annualReturns.historical.type === 'fixed')
    const { stocks, bonds } = params.plan.advanced.annualReturns.historical
    const [stocksStr, setStocksStr] = useState((stocks * 100).toFixed(1))
    const [bondsStr, setBondsStr] = useState((bonds * 100).toFixed(1))
    useEffect(() => {
      setStocksStr((stocks * 100).toFixed(1))
    }, [stocks])
    useEffect(() => {
      setBondsStr((bonds * 100).toFixed(1))
    }, [bonds])
    const handleStockAmount = (x: number) => {
      if (isNaN(x)) return
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        assert(clone.advanced.annualReturns.historical.type === 'fixed')
        clone.advanced.annualReturns.historical.stocks = x / 100
        return clone
      })
    }
    const handleBondAmount = (x: number) => {
      if (isNaN(x)) return
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        assert(clone.advanced.annualReturns.historical.type === 'fixed')
        clone.advanced.annualReturns.historical.bonds = x / 100
        return clone
      })
    }
    return (
      <div className={`${className}`}>
        <div
          className=" inline-grid  items-stretch gap-x-4 gap-y-2"
          style={{ grid: 'auto/auto 80px auto auto' }}
        >
          <h2 className="self-center">Stocks</h2>
          <input
            type="text"
            pattern="[0-9]"
            inputMode="numeric"
            className=" bg-gray-200 rounded-lg py-1.5 px-2 "
            value={stocksStr}
            onChange={(x) => setStocksStr(x.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleStockAmount(parseFloat(stocksStr))
            }}
            onBlur={(e) => handleStockAmount(parseFloat(e.target.value))}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleStockAmount(stocks * 100 + _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleStockAmount(stocks * 100 - _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
          <h2 className="self-center">Bonds</h2>
          <input
            type="text"
            pattern="[0-9]"
            inputMode="numeric"
            className=" bg-gray-200 rounded-lg py-1.5 px-2 "
            value={bondsStr}
            onChange={(x) => setBondsStr(x.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleBondAmount(parseFloat(bondsStr))
            }}
            onBlur={(e) => handleBondAmount(parseFloat(e.target.value))}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleBondAmount(bonds * 100 + _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleBondAmount(bonds * 100 - _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  },
)

const _MiscCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {
      params,
      setNonPlanParams,
      setPlanParams,
      tpawResult,
      defaultParams,
    } = useSimulation()
    const getPlanChartURL = useGetPlanChartURL()
    const defaultShowAllMonths = defaultParams.nonPlan.dev.alwaysShowAllMonths
    const isShowAllMonthsModified =
      params.nonPlan.dev.alwaysShowAllMonths !== defaultShowAllMonths

    const handleChangeShowAllMonths = (x: boolean) =>
      setNonPlanParams((nonPlan) => {
        const clone = _.cloneDeep(nonPlan)
        clone.dev.alwaysShowAllMonths = x
        return clone
      })

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge
          show={isShowAllMonthsModified}
          mainPage={false}
        />
        <h2 className="font-bold text-lg"> Misc</h2>
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <h2 className=""> Show All Months</h2>
          <ToggleSwitch
            className=""
            enabled={params.nonPlan.dev.alwaysShowAllMonths}
            setEnabled={(x) => handleChangeShowAllMonths(x)}
          />
        </div>

        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Number of simulations</h2>
          <AmountInput
            className="text-input"
            value={params.plan.advanced.monteCarloSampling.numOfSimulations}
            onChange={(numOfSimulations) =>
              setPlanParams((plan) => {
                const clone = _.cloneDeep(plan)
                clone.advanced.monteCarloSampling.numOfSimulations =
                  numOfSimulations
                return clone
              })
            }
            decimals={0}
            modalLabel="Number of Simulations"
          />
        </div>

        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Time To Run:</h2>
          <h2 className="ml-2">{`${Math.round(tpawResult.perf.main[6][1])}ms (${
            tpawResult.perf.main[6][0]
          })`}</h2>
        </div>
        <div className="mt-2">
          <h2 className="mb-1">Average Sampled Annual Returns:</h2>
          <div className="ml-4 flex gap-x-4 items-center">
            <h2 className="">Stocks</h2>
            <h2 className="ml-2">
              {formatPercentage(5)(tpawResult.averageAnnualReturns.stocks)}
            </h2>
          </div>
          <div className="ml-4 flex gap-x-4 items-center">
            <h2 className="">Bonds</h2>
            <h2 className="ml-2">
              {formatPercentage(5)(tpawResult.averageAnnualReturns.bonds)}
            </h2>
          </div>
        </div>

        <Link
          className="block underline pt-4"
          href={getPlanChartURL('asset-allocation-total-portfolio')}
          shallow
        >
          Show Asset Allocation of Total Portfolio Graph
        </Link>

        <button
          className="underline pt-4"
          onClick={async () => {
            await clearMemoizedRandom()
            setPlanParams((plan) => _.cloneDeep(plan))
          }}
        >
          Reset random draws
        </button>
        <button className="block btn-sm btn-outline mt-4" onClick={() => {}}>
          Test
        </button>
        <button
          className="mt-6 underline disabled:lighten-2 block"
          onClick={() => handleChangeShowAllMonths(defaultShowAllMonths)}
          disabled={!isShowAllMonthsModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _AdditionalSpendingTiltCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setPlanParams, defaultParams } = useSimulation()
    const defaultRisk = defaultParams.plan.risk.tpaw
    const isModified =
      defaultRisk.additionalAnnualSpendingTilt !==
      params.plan.risk.tpaw.additionalAnnualSpendingTilt

    const handleChange = (value: number) =>
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        clone.risk.tpaw.additionalAnnualSpendingTilt = value
        return clone
      })
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg">Additional Spending Tilt</h2>
        <p className="p-base mt-2">
          This lets you shift your spending between early and late retirement.
          To spend more in early retirement and less in late retirement, move
          the slider to the left. To spend more in late retirement and less in
          early retirement, move the slider to the right.
        </p>

        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES}
          value={params.plan.risk.tpaw.additionalAnnualSpendingTilt}
          onChange={(x) => handleChange(x)}
          format={(x) => formatPercentage(1)(x)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() => handleChange(defaultRisk.additionalAnnualSpendingTilt)}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _TimeCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { currentTime, params, setNonPlanParams, paramsExt, resetParams } =
      useSimulation()
    const { getDateTimeInCurrentTimezone } = paramsExt
    const { portfolioBalance } = params.plan.wealth
    const portfolioBalanceTimestamp = portfolioBalance.isLastPlanChange
      ? portfolioBalance.timestamp
      : portfolioBalance.original.timestamp

    const defaultParams = useMemo(
      () => getDefaultPlanParams(currentTime),
      [currentTime],
    )

    const isModified = _isTimeCardModified(params, defaultParams)

    // const handleChange = (x: number) =>
    //   setParams((params) => {
    //     const clone = _.cloneDeep(params)
    //     clone.dev.currentTimeMonthOffset = x
    //     return clone
    //   })

    const formatDateTime = (x: DateTime) => x.toFormat('LL/d/yyyy - HH:mm EEE')
    const formatDistanceFromNow = (x: DateTime | number) => {
      const diff = currentTime.diff(DateTime.fromMillis(x.valueOf()), [
        'years',
        'months',
        'days',
        'hours',
      ])
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
        <h2 className="font-bold text-lg">Time</h2>
        <h2 className="-mt-2 text-sm text-right">
          Zone: {currentTime.toFormat('ZZZZ')}
        </h2>
        <div
          className="grid mt-2 gap-x-4 gap-y-2"
          style={{ grid: 'auto/auto 1fr' }}
        >
          <h2 className="text-right">Current</h2>
          <h2 className="font-mono text-sm">{formatDateTime(currentTime)}</h2>
          <h2 className="text-right">Params</h2>
          <div className="">
            <h2 className="font-mono text-sm">
              {formatDateTime(
                getDateTimeInCurrentTimezone.fromMillis(params.plan.timestamp),
              )}
            </h2>
            <h2 className="text-sm font-font1 lighten">
              {formatDistanceFromNow(params.plan.timestamp)} ago
            </h2>
          </div>
          <h2 className="text-right">Portfolio</h2>
          <div className="">
            <h2 className="font-mono text-sm">
              {formatDateTime(
                getDateTimeInCurrentTimezone.fromMillis(
                  portfolioBalanceTimestamp,
                ),
              )}
            </h2>
            <h2 className="text-sm font-font1 lighten">
              {formatDistanceFromNow(portfolioBalanceTimestamp)} ago
            </h2>
          </div>
        </div>
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <h2 className="font-semibold"> Fast Forward</h2>
          <ToggleSwitch
            className=""
            enabled={
              params.nonPlan.dev.currentTimeFastForward.shouldFastForward
            }
            setEnabled={(enabled) => {
              if (enabled) {
                setNonPlanParams((nonPlan) => {
                  const clone = _.cloneDeep(nonPlan)
                  clone.dev.currentTimeFastForward = {
                    shouldFastForward: true,
                    restoreTo: JSON.stringify(params),
                    years: 0,
                    months: 0,
                    days: 0,
                    hours: 0,
                    marketDataExtensionStrategy: {
                      dailyStockMarketPerformance: 'roundRobinPastValues',
                    },
                  }
                  return clone
                })
              } else {
                assert(
                  params.nonPlan.dev.currentTimeFastForward.shouldFastForward,
                )
                const clone = planParamsGuard(
                  JSON.parse(
                    params.nonPlan.dev.currentTimeFastForward.restoreTo,
                  ),
                ).force()
                assert(
                  !clone.nonPlan.dev.currentTimeFastForward.shouldFastForward,
                )
                resetParams(clone)
              }
            }}
          />
        </div>
        {params.nonPlan.dev.currentTimeFastForward.shouldFastForward && (
          <div className="ml-4">
            <div
              className="items-center grid gap-x-2 gap-y-2 mt-4 "
              style={{ grid: 'auto/auto 1fr' }}
            >
              <_FastForwardInput type="years" />
              <_FastForwardInput type="months" />
              <_FastForwardInput type="days" />
              <_FastForwardInput type="hours" />
            </div>
            <h2 className="mt-4 font-semibold">
              How to synthesize market data?
            </h2>
            <h2 className="font-medium mt-2">
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
            <h2 className="font-medium mt-4 mb-2">
              Daily VT and BND Performance{' '}
            </h2>
            <RadioGroup
              value={
                params.nonPlan.dev.currentTimeFastForward
                  .marketDataExtensionStrategy.dailyStockMarketPerformance
              }
              onChange={(type) =>
                setNonPlanParams((nonPlan) => {
                  const clone = _.cloneDeep(nonPlan)
                  assert(clone.dev.currentTimeFastForward.shouldFastForward)
                  clone.dev.currentTimeFastForward.marketDataExtensionStrategy.dailyStockMarketPerformance =
                    type
                  return clone
                })
              }
              className="grid gap-y-2"
            >
              <RadioGroup.Option value="latestExpected">
                {({ checked }) => (
                  <div className="flex items-start gap-x-2 cursor-pointer">
                    <FontAwesomeIcon
                      className="text-sm mt-1.5"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />
                    <div className="">
                      <h2 className="">Latest expected value</h2>
                      {checked ? (
                        params.plan.advanced.annualReturns.expected.type ===
                        'manual' ? (
                          <h2 className="">Expected return is manual.</h2>
                        ) : (
                          <h2 className="text-errorFG">
                            Expected return is NOT manual.
                          </h2>
                        )
                      ) : (
                        false
                      )}
                    </div>
                  </div>
                )}
              </RadioGroup.Option>
              <RadioGroup.Option value="roundRobinPastValues">
                {({ checked }) => (
                  <div className="flex items-start gap-x-2 cursor-pointer">
                    <FontAwesomeIcon
                      className="text-sm mt-1.5"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />
                    <h2 className="">
                      Round robin of last 30 days before fast forward.
                    </h2>
                  </div>
                )}
              </RadioGroup.Option>
              <RadioGroup.Option value="repeatGrowShrinkZero">
                {({ checked }) => (
                  <div className="flex items-start gap-x-2 cursor-pointer">
                    <FontAwesomeIcon
                      className="text-sm mt-1.5"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />
                    <h2 className="">Cycle through grow (5%), shrink, flat</h2>
                  </div>
                )}
              </RadioGroup.Option>
            </RadioGroup>
          </div>
        )}
      </div>
    )
  },
)

const _FastForwardInput = React.memo(
  ({ type }: { type: 'years' | 'months' | 'days' | 'hours' }) => {
    const { params, setNonPlanParams } = useSimulation()
    assert(params.nonPlan.dev.currentTimeFastForward.shouldFastForward)
    return (
      <>
        <h2 className="">{capitalize(type)}</h2>
        <NumberInput
          className=""
          showDecrement={false}
          value={params.nonPlan.dev.currentTimeFastForward[type]}
          setValue={(x: number) => {
            const clone = _.cloneDeep(params.nonPlan)
            assert(clone.dev.currentTimeFastForward.shouldFastForward)
            const clamped = Math.max(x, clone.dev.currentTimeFastForward[type])
            clone.dev.currentTimeFastForward[type] = clamped
            setNonPlanParams(clone)
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

const _isTimeCardModified = (params: Params, defaultParams: Params) =>
  !_.isEqual(
    params.nonPlan.dev.currentTimeFastForward,
    defaultParams.nonPlan.dev.currentTimeFastForward,
  )
