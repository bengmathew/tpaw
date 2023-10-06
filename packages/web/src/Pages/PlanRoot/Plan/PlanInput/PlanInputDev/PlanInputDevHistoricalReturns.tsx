import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import {
  faCircle as faCircleSelected,
  faMinus,
  faPlus,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams, noCase } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useState } from 'react'
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
      adjustExpectedReturn: PlanParams['advanced']['historicalReturnsAdjustment']['stocks']['adjustExpectedReturn'],
    ) =>
      updatePlanParams('setHistoricalReturnsAdjustExpectedReturnDev', {
        type,
        adjustExpectedReturn,
      })

    const currValue =
      planParams.advanced.historicalReturnsAdjustment[type].adjustExpectedReturn

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <h2 className="font-bold text-lg">
          Expected Value Adjustment for {_.capitalize(type)}
        </h2>

        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'toExpectedUsedForPlanning',
                correctForBlockSampling: true,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'toExpectedUsedForPlanning'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjusted to Expected Used for Planning
          </h2>
          {currValue.type === 'toExpectedUsedForPlanning' && (
            <_CorrectForBlockSamplingInput
              className="ml-[28px] mt-2"
              curr={currValue.correctForBlockSampling}
              onChange={(correctForBlockSampling) =>
                handleChange({
                  type: 'toExpectedUsedForPlanning',
                  correctForBlockSampling,
                })
              }
            />
          )}
        </div>

        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() =>
              handleChange({
                type: 'toAnnualExpectedReturn',
                annualExpectedReturn:
                  planParamsProcessed.expectedReturnsForPlanning.annual[type],
                correctForBlockSampling: true,
              })
            }
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'toAnnualExpectedReturn'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            Adjust to Other
          </h2>
          {currValue.type === 'toAnnualExpectedReturn' && (
            <div className="">
              <_ExpectedReturnInput
                className="ml-[28px] mt-2"
                curr={currValue.annualExpectedReturn}
                onChange={(value) =>
                  handleChange({ ...currValue, annualExpectedReturn: value })
                }
              />
              <_CorrectForBlockSamplingInput
                curr={currValue.correctForBlockSampling}
                onChange={(correctForBlockSampling) =>
                  handleChange({ ...currValue, correctForBlockSampling })
                }
              />
            </div>
          )}
        </div>
        <div className="mt-4">
          <h2
            className={`cursor-pointer `}
            onClick={() => handleChange({ type: 'none' })}
          >
            <FontAwesomeIcon
              className="mr-2"
              icon={
                currValue.type === 'none' ? faCircleSelected : faCircleRegular
              }
            />{' '}
            Do Not Adjust
          </h2>
        </div>
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.historicalReturnsAdjustment[type]
                .adjustExpectedReturn,
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

const _CorrectForBlockSamplingInput = React.memo(
  ({
    className = '',
    curr,
    onChange,
  }: {
    className?: string
    curr: boolean
    onChange: (value: boolean) => void
  }) => (
    <div className={`${className} ml-8 `}>
      <div className="flex  items-center gap-x-4  py-1.5">
        <h2 className="">Correct for Block Sampling</h2>
        <ToggleSwitch
          className=""
          checked={curr}
          setChecked={(x) => onChange(x)}
        />
      </div>
    </div>
  ),
)

const _ExpectedReturnInput = React.memo(
  ({
    className = '',
    curr,
    onChange,
  }: {
    className?: string
    curr: number
    onChange: (value: number) => void
  }) => {
    const _delta = 0.1
    const [str, setStr] = useState((curr * 100).toFixed(1))
    useEffect(() => {
      setStr((curr * 100).toFixed(1))
    }, [curr])
    const handleAmount = (x: number) => {
      if (isNaN(x)) return
      onChange(_.round(x / 100, getPrecision(x) + 2))
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
            onClick={() => handleAmount(curr * 100 + _delta)}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(curr * 100 - _delta)}
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
  return !_.isEqual(
    planParams.advanced.historicalReturnsAdjustment[type].adjustExpectedReturn,
    defaultPlanParams.advanced.historicalReturnsAdjustment[type]
      .adjustExpectedReturn,
  )
}

export const PlanInputDevHistoricalReturnsSummary = React.memo(() => {
  const { planParams } = useSimulation()
  const byType = (type: 'stocks' | 'bonds') => {
    const curr =
      planParams.advanced.historicalReturnsAdjustment[type].adjustExpectedReturn
    switch (curr.type) {
      case 'none':
        return <h2>None</h2>
      case 'toExpectedUsedForPlanning':
        return (
          <>
            <h2>Adjusted to Expected</h2>
            <h2>
              Corrected for Block Sampling:{' '}
              {curr.correctForBlockSampling ? 'true' : 'false'}
            </h2>
          </>
        )
      case 'toAnnualExpectedReturn':
        return (
          <>
            <h2>
              Adjusted to {formatPercentage(2)(curr.annualExpectedReturn)}
            </h2>
            <h2>
              Corrected for Block Sampling:
              {curr.correctForBlockSampling ? 'true' : 'false'}
            </h2>
          </>
        )
      default:
        noCase(curr)
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
