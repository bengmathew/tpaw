import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  getDefaultPlanParams,
  MANUAL_INFLATION_VALUES,
  PlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { processInflation } from '../../../TPAWSimulator/PlanParamsProcessed'
import { Contentful } from '../../../Utils/Contentful'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../Utils/Geometry'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputInflation = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <_InflationCard className="" props={props} />
      </PlanInputBody>
    )
  },
)

export const _InflationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const marketData = useMarketData()
    const content = usePlanContent().inflation.intro[params.strategy]

    const handleChange = (inflation: PlanParams['inflation']) => {
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.inflation = inflation
        return clone
      })
    }
    return (
      <div
        className={`${className} params-card`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <Contentful.RichText body={content} p="col-span-2 mb-2 p-base" />

        <button
          className={`${className} flex gap-x-2 mt-4`}
          onClick={() => handleChange({ type: 'suggested' })}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.inflation.type === 'suggested'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({ type: 'suggested' })}
            </h2>
            <h2 className="text-left text-sm lighten-2">
              {formatPercentage(1)(
                processInflation({ type: 'suggested' }, marketData),
              )}
            </h2>
          </div>
        </button>

        <button
          className={`${className} flex gap-x-2 mt-3`}
          onClick={() =>
            handleChange({
              type: 'manual',
              value: processInflation(params.inflation, marketData),
            })
          }
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.inflation.type === 'manual'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({ type: 'manual' })}
            </h2>
          </div>
        </button>
        {params.inflation.type === 'manual' && (
          <SliderInput
            className=""
            height={60}
            maxOverflowHorz={props.sizing.cardPadding}
            format={formatPercentage(1)}
            data={MANUAL_INFLATION_VALUES}
            value={params.inflation.value}
            onChange={(value) => handleChange({ type: 'manual', value })}
            ticks={(value, i) =>
              i % 10 === 0
                ? ('large' as const)
                : i % 2 === 0
                ? ('small' as const)
                : ('none' as const)
            }
          />
        )}
        <button
          className="mt-6 underline"
          onClick={() => handleChange(getDefaultPlanParams().inflation)}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const inflationTypeLabel = ({
  type,
}: {
  type: PlanParams['inflation']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Suggested'
    case 'manual':
      return 'Manual'
  }
}
