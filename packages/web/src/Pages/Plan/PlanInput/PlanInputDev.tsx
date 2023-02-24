import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import {
  faCircle as faCircleSelected,
  faMinus,
  faPlus,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES,
  annualToMonthlyReturnRate,
  assert,
  getDefaultPlanParams,
  getStats,
  historicalReturns,
  PlanParams,
  sequentialAnnualReturnsFromMonthly,
} from '@tpaw/common'
import _ from 'lodash'
import Link from 'next/link'
import React, { useEffect, useMemo, useState } from 'react'
import {
  generateSampledAnnualReturnStatsTable,
  getAnnualToMonthlyRateConvertionCorrection,
} from '../../../TPAWSimulator/PlanParamsProcessed/GetAnnualToMonthlyRateConvertionCorrection'
import { clearMemoizedRandom } from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
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
    const { params, setParams, paramsProcessed } = useSimulation()

    const defaultHistorical = useMemo(
      () => getDefaultPlanParams().advanced.annualReturns.historical,
      [],
    )
    const isModified = !_.isEqual(
      defaultHistorical,
      params.advanced.annualReturns.historical,
    )

    const handleChange = (
      historical: PlanParams['advanced']['annualReturns']['historical'],
    ) => {
      setParams((params) => {
        const clone = _.cloneDeep(params)
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
                params.advanced.annualReturns.historical.type === 'adjusted'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjusted to Expected
          </h2>
          {params.advanced.annualReturns.historical.type === 'adjusted' && (
            <_HistoricalAdjustedDetails
              className="ml-[28px] mt-2"
              historical={params.advanced.annualReturns.historical}
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
                params.advanced.annualReturns.historical.type === 'unadjusted'
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
                params.advanced.annualReturns.historical.type === 'fixed'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Fixed
          </h2>
          {params.advanced.annualReturns.historical.type === 'fixed' && (
            <_HistoricalFixedDetails className="ml-[28px] mt-2" />
          )}
        </div>

        <h2 className="font-semibold mt-6">
          Adjustment Corrections for Block Sampling
        </h2>

        <div className="ml-4 flex gap-x-4">
          <button
            className="mt-4 underline"
            onClick={async () =>
              console.dir(await generateSampledAnnualReturnStatsTable())
            }
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
    const { setParams } = useSimulation()
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
              setParams((p) => {
                const clone = _.cloneDeep(p)
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
    const { params, setParams } = useSimulation()
    assert(params.advanced.annualReturns.historical.type === 'fixed')
    const { stocks, bonds } = params.advanced.annualReturns.historical
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
      setParams((p) => {
        const clone = _.cloneDeep(p)
        assert(clone.advanced.annualReturns.historical.type === 'fixed')
        clone.advanced.annualReturns.historical.stocks = x / 100
        return clone
      })
    }
    const handleBondAmount = (x: number) => {
      if (isNaN(x)) return
      setParams((p) => {
        const clone = _.cloneDeep(p)
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
    const { params, setParams, numRuns, setNumRuns, tpawResult } =
      useSimulation()
    const getPlanChartURL = useGetPlanChartURL()
    const defaultShowAllMonths = useMemo(
      () => getDefaultPlanParams().dev.alwaysShowAllMonths,
      [],
    )
    const isShowAllMonthsModified =
      params.dev.alwaysShowAllMonths !== defaultShowAllMonths

    const handleChangeShowAllMonths = (x: boolean) =>
      setParams((p) => {
        const clone = _.cloneDeep(p)
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
            enabled={params.dev.alwaysShowAllMonths}
            setEnabled={(x) => handleChangeShowAllMonths(x)}
          />
        </div>

        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Number of simulations</h2>
          <AmountInput
            className="text-input"
            value={numRuns}
            onChange={setNumRuns}
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
          href={getPlanChartURL('asset-allocation-total-portfolio')}
          shallow
        >
          <a className="block underline pt-4">
            Show Asset Allocation of Total Portfolio Graph
          </a>
        </Link>

        <button
          className="underline pt-4"
          onClick={async () => {
            await clearMemoizedRandom()
            setParams((x) => _.cloneDeep(x))
          }}
        >
          Reset random draws
        </button>
        <button
          className="block btn-sm btn-outline mt-4"
          onClick={() => {
            const stocks = historicalReturns.monthly.stocks

            const getAnnualMean = (monthly: number[]) =>
              getStats(sequentialAnnualReturnsFromMonthly(monthly)).mean

            const targetAnnual = getAnnualMean(stocks.returns)
            const targetMonthly = annualToMonthlyReturnRate(targetAnnual)
            const meanWithoutCorrection = getAnnualMean(
              stocks.adjust(targetMonthly),
            )
            const meanWithCorrection = getAnnualMean(
              stocks.adjust(
                targetMonthly -
                  getAnnualToMonthlyRateConvertionCorrection.forHistoricalSequence(
                    'stocks',
                  ),
              ),
            )
            const meanWithCorrection2 = getAnnualMean(
              stocks.adjust(
                targetMonthly -
                  getAnnualToMonthlyRateConvertionCorrection.forMonteCarlo(
                    12,
                    'stocks',
                  ),
              ),
            )
            console.dir(
              `Mean without correction: ${Math.abs(
                targetAnnual - meanWithoutCorrection,
              )}`,
            )
            console.dir(
              `   Mean with correction: ${Math.abs(
                targetAnnual - meanWithCorrection,
              )}`,
            )
            console.dir(
              `  Mean with correction2: ${Math.abs(
                targetAnnual - meanWithCorrection2,
              )}`,
            )
          }}
        >
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
    const { params, setParams } = useSimulation()
    const defaultRisk = useMemo(() => getDefaultPlanParams().risk.tpaw, [])
    const isModified =
      defaultRisk.additionalAnnualSpendingTilt !==
      params.risk.tpaw.additionalAnnualSpendingTilt

    const handleChange = (value: number) =>
      setParams((params) => {
        const clone = _.cloneDeep(params)
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
          value={params.risk.tpaw.additionalAnnualSpendingTilt}
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
