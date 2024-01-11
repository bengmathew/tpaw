import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  MANUAL_INFLATION_VALUES,
  noCase,
  PlanParams,
  SUGGESTED_ANNUAL_INFLATION,
} from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { PlanParamsProcessed } from '../../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../Utils/Geometry'
import { SliderInput } from '../../../Common/Inputs/SliderInput/SliderInput'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
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
    const {
      planParams,
      updatePlanParams,
      defaultPlanParams,
      currentMarketData,
    } = useSimulation()

    const handleChange = (
      annualInflation: PlanParams['advanced']['annualInflation'],
    ) => updatePlanParams('setAnnualInflation', annualInflation)

    const isModified = useIsPlanInputInflationModified()

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
              planParams.advanced.annualInflation.type === 'suggested'
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
                SUGGESTED_ANNUAL_INFLATION(currentMarketData),
              )}
            </h2>
          </div>
        </button>

        <button
          className={`${className} flex gap-x-2 mt-3`}
          onClick={() => {
            switch (planParams.advanced.annualInflation.type) {
              case 'suggested':
                handleChange({
                  type: 'manual',
                  value: SUGGESTED_ANNUAL_INFLATION(currentMarketData),
                })
                break
              case 'manual':
                return
              default:
                noCase(planParams.advanced.annualInflation)
            }
          }}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              planParams.advanced.annualInflation.type === 'manual'
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
        {planParams.advanced.annualInflation.type === 'manual' && (
          <SliderInput
            className=""
            height={60}
            maxOverflowHorz={props.sizing.cardPadding}
            format={formatPercentage(1)}
            data={MANUAL_INFLATION_VALUES}
            value={planParams.advanced.annualInflation.value}
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
          onClick={() =>
            handleChange(defaultPlanParams.advanced.annualInflation)
          }
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

export const useIsPlanInputInflationModified = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.annualInflation,
    planParams.advanced.annualInflation,
  )
}

export const PlanInputInflationSummary = React.memo(
  ({ planParamsProcessed }: { planParamsProcessed: PlanParamsProcessed }) => {
    const { planParams } = planParamsProcessed
    const format = formatPercentage(1)
    return (
      <h2>
        {inflationTypeLabel(planParams.advanced.annualInflation)}:{' '}
        {format(planParamsProcessed.inflation.annual)}
      </h2>
    )
  },
)
