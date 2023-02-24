import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  getDefaultPlanParams,
  MANUAL_INFLATION_VALUES,
  noCase,
  PlanParams,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../Utils/Geometry'
import { useMarketData } from '../../App/WithMarketData'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { usePlanContent } from '../Plan'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
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
    const content = usePlanContent().inflation.intro[params.advanced.strategy]

    const handleChange = (
      annualInflation: PlanParams['advanced']['annualInflation'],
    ) => {
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.advanced.annualInflation = annualInflation
        return clone
      })
    }

    const defaultValue = getDefaultPlanParams().advanced.annualInflation
    const isModified = !_.isEqual(defaultValue, params.advanced.annualInflation)

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <p className="mb-2 p-base mt-1">
          {`Enter your estimate for the annual inflation rate. The "suggested" option will be automatically updated periodically based on new data.`}
        </p>

        <button
          className={`${className} flex gap-x-2 mt-4`}
          onClick={() => handleChange({ type: 'suggested' })}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.advanced.annualInflation.type === 'suggested'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({ type: 'suggested' })}
            </h2>
            <h2 className="text-left text-sm lighten-2">
              {formatPercentage(1)(SUGGESTED_ANNUAL_INFLATION(marketData))}
            </h2>
          </div>
        </button>

        <button
          className={`${className} flex gap-x-2 mt-3`}
          onClick={() => {
            switch (params.advanced.annualInflation.type) {
              case 'suggested':
                handleChange({
                  type: 'manual',
                  value: SUGGESTED_ANNUAL_INFLATION(marketData),
                })
                break
              case 'manual':
                return
              default:
                noCase(params.advanced.annualInflation)
            }
          }}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.advanced.annualInflation.type === 'manual'
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
        {params.advanced.annualInflation.type === 'manual' && (
          <SliderInput
            className=""
            height={60}
            maxOverflowHorz={props.sizing.cardPadding}
            format={formatPercentage(1)}
            data={MANUAL_INFLATION_VALUES}
            value={params.advanced.annualInflation.value}
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
          className="mt-6 underline disabled:lighten-2"
          onClick={() => handleChange(defaultValue)}
          disabled={!isModified}
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
  type: PlanParams['advanced']['annualInflation']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Suggested'
    case 'manual':
      return 'Manual'
  }
}
