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
        <>
          <_HistoricalReturnsCard
            className="mt-10"
            props={props}
            type="stocks"
          />
          <_HistoricalReturnsCard
            className="mt-10"
            props={props}
            type="bonds"
          />
          <_AdjustmentCorrectionsCard className="mt-10" props={props} />
        </>
      </PlanInputBody>
    )
  },
)
const _HistoricalReturnsCard = React.memo(
  ({
    className = '',
    props,
    type,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    type: 'stocks' | 'bonds'
  }) => {
    const {
      planParams,
      updatePlanParams,
      planParamsProcessed,
      defaultPlanParams,
    } = useSimulation()

    const isModified = useIsCardModified(type)

    const handleChange = (
      value: PlanParams['advanced']['annualReturns']['historical']['stocks'],
    ) =>
      type === 'stocks'
        ? updatePlanParams('setHistoricalReturnsStocksDev', value)
        : type === 'bonds'
        ? updatePlanParams('setHistoricalReturnsBondsDev', value)
        : noCase(type)

    const currValue = planParams.advanced.annualReturns.historical[type]

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <h2 className="font-bold text-lg">{_.capitalize(type)}</h2>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'adjustExpected',
                adjustment: { type: 'toExpectedUsedForPlanning' },
                correctForBlockSampling: true,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'adjustExpected'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjusted to Expected
          </h2>
          {currValue.type === 'adjustExpected' && (
            <_HistoricalAdjustedDetails
              className="ml-[28px] mt-2"
              historical={currValue}
              onChange={handleChange}
            />
          )}
        </div>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() => handleChange({ type: 'rawHistorical' })}
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'rawHistorical'
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
                value: { type: 'expectedUsedForPlanning' },
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'fixed' &&
                currValue.value.type === 'expectedUsedForPlanning'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Fixed to Expected
          </h2>
        </div>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'fixed',
                value: {
                  type: 'manual',
                  value:
                    planParamsProcessed.returns.expectedAnnualReturns[type],
                },
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'fixed' && currValue.value.type === 'manual'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Fixed to Manual
          </h2>
          {currValue.type === 'fixed' && currValue.value.type === 'manual' && (
            <_HistoricalFixedToManualDetails
              className="ml-[28px] mt-2"
              type={type}
              onChange={handleChange}
            />
          )}
        </div>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.annualReturns.historical[type],
            )
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
    onChange,
  }: {
    className?: string
    historical: Extract<
      PlanParams['advanced']['annualReturns']['historical']['stocks'],
      { type: 'adjustExpected' }
    >
    onChange: (
      value: PlanParams['advanced']['annualReturns']['historical']['stocks'],
    ) => void
  }) => {
    const { adjustment, correctForBlockSampling } = historical
    assert(adjustment.type === 'toExpectedUsedForPlanning')
    return (
      <div className={`${className} ml-8 `}>
        <div className="flex  items-center gap-x-4  py-1.5">
          <h2 className="">Correct for Block Sampling</h2>
          <ToggleSwitch
            className=""
            checked={correctForBlockSampling}
            setChecked={(x) =>
              onChange({ ...historical, correctForBlockSampling: x })
            }
          />
        </div>
      </div>
    )
  },
)
const _HistoricalFixedToManualDetails = React.memo(
  ({
    className = '',
    type,
    onChange,
  }: {
    className?: string
    type: 'stocks' | 'bonds'
    onChange: (
      value: PlanParams['advanced']['annualReturns']['historical']['stocks'],
    ) => void
  }) => {
    const _delta = 0.1
    const { planParams } = useSimulation()
    const currHistorical = planParams.advanced.annualReturns.historical[type]
    assert(currHistorical.type === 'fixed')
    assert(currHistorical.value.type === 'manual')
    const currValue = currHistorical.value.value
    const [str, setStr] = useState((currValue * 100).toFixed(1))
    useEffect(() => {
      setStr((currValue * 100).toFixed(1))
    }, [currValue])
    const handleAmount = (x: number) => {
      if (isNaN(x)) return
      onChange({
        type: 'fixed',
        value: { type: 'manual', value: _.round(x / 100, getPrecision(x) + 2) },
      })
    }
    return (
      <div className={`${className}`}>
        <div
          className=" inline-grid  items-stretch gap-x-4 gap-y-2"
          style={{ grid: 'auto/ 80px auto auto' }}
        >
          <input
            type="text"
            pattern="[0-9]"
            inputMode="numeric"
            className=" bg-gray-200 rounded-lg py-1.5 px-2 "
            value={str}
            onChange={(x) => setStr(x.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAmount(parseFloat(str))
            }}
            onBlur={(e) => handleAmount(parseFloat(e.target.value))}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(currValue * 100 + _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(currValue * 100 - _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  },
)

const _AdjustmentCorrectionsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">
          Adjustment Corrections for Block Sampling
        </h2>
        <div className="flex gap-x-4">
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
      </div>
    )
  },
)

export const useIsPlanInputDevHistoricalReturnsModified = () => {
  const stocksModified = useIsCardModified('stocks')
  const bondsModified = useIsCardModified('bonds')
  return stocksModified || bondsModified
}

const useIsCardModified = (type: 'stocks' | 'bonds') => {
  const { planParams, defaultPlanParams } = useSimulation()
  return useMemo(
    () =>
      !_.isEqual(
        planParams.advanced.annualReturns.historical[type],
        defaultPlanParams.advanced.annualReturns.historical[type],
      ),
    [
      defaultPlanParams.advanced.annualReturns.historical,
      planParams.advanced.annualReturns.historical,
      type,
    ],
  )
}

export const PlanInputDevHistoricalReturnsSummary = React.memo(() => {
  const { planParams } = useSimulation()
  const byType = (type: 'stocks' | 'bonds') => {
    const historical = planParams.advanced.annualReturns.historical[type]
    switch (historical.type) {
      case 'adjustExpected': {
        switch (historical.adjustment.type) {
          case 'toExpectedUsedForPlanning':
            return (
              <>
                <h2>Adjusted to Expected</h2>
                <h2>
                  Corrected for Block Sampling:{' '}
                  {historical.correctForBlockSampling}
                </h2>
              </>
            )
          case 'byValue':
          case 'toValue':
            return (
              <>
                <h2>
                  Adjusted {historical.adjustment.type.slice(2)}:{' '}
                  {formatPercentage(2)(historical.adjustment.value)}
                </h2>
                <h2>
                  Corrected for Block Sampling:{' '}
                  {historical.correctForBlockSampling}
                </h2>
              </>
            )
          default:
            noCase(historical.adjustment)
        }
      }
      case 'rawHistorical':
        return <h2>Unadjusted</h2>
      case 'fixed':
        return historical.value.type === 'manual' ? (
          <>Fixed to {formatPercentage(2)(historical.value.value)}</>
        ) : historical.value.type === 'expectedUsedForPlanning' ? (
          <>Fixed to Expected</>
        ) : (
          noCase(historical.value)
        )
      default:
        noCase(historical)
    }
  }

  return (
    <>
      <h2>Stocks</h2>
      <div className="ml-4">{byType('stocks')}</div>
      <h2>Bonds</h2>
      <div className="ml-4">{byType('bonds')}</div>
    </>
  )
})
