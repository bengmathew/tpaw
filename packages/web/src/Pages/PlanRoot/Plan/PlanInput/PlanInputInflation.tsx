import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PLAN_PARAMS_CONSTANTS, noCase, PlanParams } from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { PlanParamsProcessed } from '../../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../Utils/Geometry'
import { SliderInput } from '../../../Common/Inputs/SliderInput/SliderInput'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { useWASM } from '../../PlanRootHelpers/WithWASM'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'
import { fWASM } from '../../../../UseSimulator/Simulator/GetWASM'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'

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
      planParamsNorm,
      updatePlanParams,
      defaultPlanParams,
      currentMarketData,
    } = useSimulation()

    const suggestedInflation = useMemo(
      () => fWASM().get_suggested_annual_inflation(currentMarketData),
      [currentMarketData],
    )

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
              planParamsNorm.advanced.annualInflation.type === 'suggested'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({ type: 'suggested' })}
            </h2>
            <h2 className="text-left text-sm lighten-2">
              {formatPercentage(1)(suggestedInflation)}
            </h2>
          </div>
        </button>

        <button
          className={`${className} flex gap-x-2 mt-3`}
          onClick={() => {
            switch (planParamsNorm.advanced.annualInflation.type) {
              case 'suggested':
                handleChange({
                  type: 'manual',
                  value: suggestedInflation,
                })
                break
              case 'manual':
                return
              default:
                noCase(planParamsNorm.advanced.annualInflation)
            }
          }}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              planParamsNorm.advanced.annualInflation.type === 'manual'
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
        {planParamsNorm.advanced.annualInflation.type === 'manual' && (
          <SliderInput
            className=""
            height={60}
            maxOverflowHorz={props.sizing.cardPadding}
            format={formatPercentage(1)}
            data={PLAN_PARAMS_CONSTANTS.manualInflationValues}
            value={planParamsNorm.advanced.annualInflation.value}
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
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.annualInflation,
    planParamsNorm.advanced.annualInflation,
  )
}

export const PlanInputInflationSummary = React.memo(
  ({
    planParamsNorm,
    planParamsProcessed,
  }: {
    planParamsNorm: PlanParamsNormalized
    planParamsProcessed: PlanParamsProcessed
  }) => {
    const format = formatPercentage(1)
    return (
      <h2>
        {inflationTypeLabel(planParamsNorm.advanced.annualInflation)}:{' '}
        {format(planParamsProcessed.inflation.annual)}
      </h2>
    )
  },
)
