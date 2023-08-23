import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import {
  faCircle as faCircleSelected,
  faMinus,
  faPlus,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams, assert, noCase } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { getPrecision } from '../../../../../Utils/GetPrecision'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevHistoricalReturns = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_HistoricalReturnsCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)
const _HistoricalReturnsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams, planParamsProcessed, defaultPlanParams } =
      useSimulation()

    const isModified = useIsPlanInputDevHistoricalReturnsModified()

    const handleChange = (
      historical: PlanParams['advanced']['annualReturns']['historical'],
    ) => updatePlanParams('switchHistoricalReturns', historical)
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
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
                planParams.advanced.annualReturns.historical.type === 'adjusted'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjusted to Expected
          </h2>
          {planParams.advanced.annualReturns.historical.type === 'adjusted' && (
            <_HistoricalAdjustedDetails
              className="ml-[28px] mt-2"
              historical={planParams.advanced.annualReturns.historical}
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
                planParams.advanced.annualReturns.historical.type ===
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
                stocks: planParamsProcessed.returns.expectedAnnualReturns.stocks,
                bonds: planParamsProcessed.returns.expectedAnnualReturns.bonds,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                planParams.advanced.annualReturns.historical.type === 'fixed'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Fixed
          </h2>
          {planParams.advanced.annualReturns.historical.type === 'fixed' && (
            <_HistoricalFixedDetails className="ml-[28px] mt-2" />
          )}
        </div>

        <h2 className="font-semibold mt-6">
          Adjustment Corrections for Block Sampling
        </h2>

        <div className="ml-4 flex gap-x-4">
          <button
            className="mt-4 underline"
            // onClick={async () => {
            //   // console.dir(await generateSampledAnnualReturnStatsTable())
            // }}
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
          onClick={() =>
            handleChange(defaultPlanParams.advanced.annualReturns.historical)
          }
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
    const { updatePlanParams } = useSimulation()
    const { adjustment, correctForBlockSampling } = historical
    assert(adjustment.type === 'toExpected')
    return (
      <div className={`${className} ml-8 `}>
        <div className="flex  items-center gap-x-4  py-1.5">
          <h2 className="">Correct for Block Sampling</h2>
          <ToggleSwitch
            className=""
            checked={correctForBlockSampling}
            setChecked={(x) =>
              updatePlanParams('setHistoricalReturnsAdjustForBlockSampling', x)
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
    const { planParams, updatePlanParams } = useSimulation()
    assert(planParams.advanced.annualReturns.historical.type === 'fixed')
    const { stocks, bonds } = planParams.advanced.annualReturns.historical
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
      updatePlanParams(
        'setHistoricalReturnsFixedStocks',
        _.round(x / 100, getPrecision(x) + 2),
      )
    }
    const handleBondAmount = (x: number) => {
      if (isNaN(x)) return
      updatePlanParams(
        'setHistoricalReturnsFixedBonds',
        _.round(x / 100, getPrecision(x) + 2),
      )
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

export const useIsPlanInputDevHistoricalReturnsModified = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return useMemo(
    () =>
      !_.isEqual(
        planParams.advanced.annualReturns.historical,
        defaultPlanParams.advanced.annualReturns.historical,
      ),
    [defaultPlanParams, planParams],
  )
}

export const PlanInputDevHistoricalReturnsSummary = React.memo(() => {
  const { planParams } = useSimulation()
  const { historical } = planParams.advanced.annualReturns
  switch (historical.type) {
    case 'adjusted': {
      switch (historical.adjustment.type) {
        case 'toExpected':
          return <h2>Adjusted to Expected</h2>
        case 'by':
        case 'to':
          return (
            <>
              <h2>Adjusted {historical.adjustment.type}:</h2>
              <h2 className="ml-4">
                Stocks: {formatPercentage(2)(historical.adjustment.stocks)}
              </h2>
              <h2 className="ml-4">
                Bonds: {formatPercentage(2)(historical.adjustment.bonds)}
              </h2>
            </>
          )
        default:
          noCase(historical.adjustment)
      }
    }
    case 'unadjusted':
      return <h2>Unadjusted</h2>
    case 'fixed':
      return (
        <>
          <h2>Fixed to:</h2>
          <h2 className="ml-4">
            Stocks: {formatPercentage(2)(historical.stocks)}
          </h2>
          <h2 className="ml-4">
            Bonds: {formatPercentage(2)(historical.bonds)}
          </h2>
        </>
      )
    default:
      noCase(historical)
  }
})
