import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  EXPECTED_RETURN_PRESETS,
  getDefaultPlanParams,
  MANUAL_STOCKS_BONDS_RETURNS_VALUES,
  PlanParams
} from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { processExpectedReturns } from '../../../TPAWSimulator/PlanParamsProcessed'
import { Contentful } from '../../../Utils/Contentful'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../Utils/Geometry'
import { preciseRange } from '../../../Utils/PreciseRange'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps
} from './PlanInputBody/PlanInputBody'

export const PlanInputExpectedReturns = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <_ExpectedReturnsCard className="" props={props} />
      </PlanInputBody>
    )
  },
)

export const _ExpectedReturnsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const content = usePlanContent()

    const handleChange = (expected: PlanParams['returns']['expected']) => {
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.returns.expected = expected
        return clone
      })
    }

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <div className="">
          <Contentful.RichText
            body={content['expected-returns'].intro[params.strategy]}
            p="col-span-2 mb-2 p-base"
          />
          <_Preset className="mt-4" type="suggested" onChange={handleChange} />
          <_Preset
            className="mt-4"
            type="oneOverCAPE"
            onChange={handleChange}
          />
          <_Preset
            className="mt-4"
            type="regressionPrediction"
            onChange={handleChange}
          />
          <_Preset className="mt-4" type="historical" onChange={handleChange} />
          <_Manual className="mt-4" onChange={handleChange} props={props} />
        </div>
        <button
          className="mt-6 underline"
          onClick={() => handleChange(getDefaultPlanParams().returns.expected)}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const _Preset = React.memo(
  ({
    className = '',
    type,
    onChange,
  }: {
    className?: string
    type: Parameters<typeof EXPECTED_RETURN_PRESETS>[0]
    onChange: (expected: PlanParams['returns']['expected']) => void
  }) => {
    const { params } = useSimulation()
    const marketData = useMarketData()
    const { stocks, bonds } = EXPECTED_RETURN_PRESETS(type, marketData)

    return (
      <button
        className={`${className} flex gap-x-2`}
        onClick={() => onChange({ type })}
      >
        <FontAwesomeIcon
          className="mt-1"
          icon={
            params.returns.expected.type === type
              ? faCircleSelected
              : faCircleRegular
          }
        />
        <div className="">
          <h2 className="text-start">{expectedReturnTypeLabel({ type })}</h2>
          <h2 className="text-start lighten-2 text-sm">
            Stocks: {formatPercentage(1)(stocks)}, Bonds:{' '}
            {formatPercentage(1)(bonds)}
          </h2>
        </div>
      </button>
    )
  },
)

export const _Manual = React.memo(
  ({
    className = '',
    onChange,
    props,
  }: {
    className?: string
    onChange: (expected: PlanParams['returns']['expected']) => void
    props: PlanInputBodyPassThruProps
  }) => {
    const marketData = useMarketData()
    const { params } = useSimulation()
    const sliderProps = {
      className: '',
      height: 60,
      formatValue: (x: number) => `${(x * 100).toFixed(1)}%`,
      domain: preciseRange(-0.01, 0.1, 0.001, 3).map((value, i) => ({
        value,
        tick:
          i % 10 === 0
            ? ('large' as const)
            : i % 2 === 0
            ? ('small' as const)
            : ('none' as const),
      })),
    }
    let { stocks, bonds } = processExpectedReturns(
      params.returns.expected,
      marketData,
    )
    stocks = _.round(stocks, 3)
    bonds = _.round(bonds, 3)
    return (
      <div className={`${className}`}>
        <button
          className={`${className} flex gap-x-2`}
          onClick={() => onChange({ type: 'manual', stocks, bonds })}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.returns.expected.type === 'manual'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {expectedReturnTypeLabel({ type: 'manual' })}
            </h2>
          </div>
        </button>
        {params.returns.expected.type === 'manual' && (
          <div className="mt-4">
            <h2 className="ml-6 mt-2 ">Stocks</h2>
            <SliderInput
              className=""
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={formatPercentage(1)}
              data={MANUAL_STOCKS_BONDS_RETURNS_VALUES}
              value={params.returns.expected.stocks}
              onChange={(stocks) => onChange({ type: 'manual', stocks, bonds })}
              ticks={(value, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                  ? ('small' as const)
                  : ('none' as const)
              }
            />

            <h2 className="ml-6">Bonds</h2>
            <SliderInput
              className=""
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={formatPercentage(1)}
              data={MANUAL_STOCKS_BONDS_RETURNS_VALUES}
              value={params.returns.expected.bonds}
              onChange={(bonds) => onChange({ type: 'manual', stocks, bonds })}
              ticks={(value, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                  ? ('small' as const)
                  : ('none' as const)
              }
            />
            <p className="p-base ml-6">
              Remember to use real and not nominal returns.
            </p>
          </div>
        )}
      </div>
    )
  },
)
export const expectedReturnTypeLabel = ({
  type,
}: {
  type: PlanParams['returns']['expected']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Suggested'
    case 'oneOverCAPE':
      return '1/CAPE for stocks'
    case 'regressionPrediction':
      return 'Regression prediction for stocks'
    case 'historical':
      return 'Historical'
    case 'manual':
      return 'Manual'
  }
}
