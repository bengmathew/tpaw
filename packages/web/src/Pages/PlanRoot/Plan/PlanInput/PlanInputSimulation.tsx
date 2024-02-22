import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MAX_AGE_IN_MONTHS, PlanParams } from '@tpaw/common'
import React from 'react'
import {
    paddingCSSStyle,
    paddingCSSStyleHorz,
} from '../../../../Utils/Geometry'
import { numMonthsStr } from '../../../../Utils/NumMonthsStr'
import { NumMonthsInput } from '../../../Common/Inputs/NumMonthsInput'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
    PlanInputBody,
    PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputSimulation = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { updatePlanParams } = useSimulation()

    return (
      <PlanInputBody {...props}>
        <>
          <p
            className="p-base"
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, { scale: 0.5 }),
            }}
          >
            How should we pick sequences of returns for the simulations?
          </p>
          <_MonteCarloCard className="mt-8" props={props} />
          <_HistoricalCard className="mt-8" props={props} />
          <button
            className="mt-8 underline"
            onClick={() => updatePlanParams('setSamplingToDefault', null)}
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, { scale: 0.5 }),
            }}
          >
            Reset to Default
          </button>
        </>
      </PlanInputBody>
    )
  },
)

const _MonteCarloCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams, defaultPlanParams } = useSimulation()
    const isModified = useIsMonteCarloCardModifed()
    const isEnabled = planParams.advanced.sampling.type === 'monteCarlo'
    const handleBlockSize = (x: number) =>
      updatePlanParams('setMonteCarloSamplingBlockSize', x)

    const body = (
      <div className="">
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className="flex items-start gap-x-2 py-0.5 font-bold text-lg">
          <FontAwesomeIcon
            className="mt-1"
            icon={isEnabled ? faCircleSolid : faCircle}
          />
          <h2 className="">Monte Carlo Sequence</h2>
        </div>
        <p className="mt-2 p-base">
          Construct sequences by randomly drawing returns from the historical
          data.
        </p>
        <div className={`${isEnabled ? '' : 'lighten-2'} mt-2`}>
          <h2 className="p-base mb-2">Pick returns in blocks of:</h2>
          <NumMonthsInput
            className={` ml-8`}
            modalLabel="Sampling Block Size"
            value={planParams.advanced.sampling.forMonteCarlo.blockSize}
            onChange={handleBlockSize}
            rangeAsMFN={{ start: 1, end: MAX_AGE_IN_MONTHS }}
            disabled={!isEnabled}
          />
          <button
            className="mt-4 underline disabled:lighten-2"
            disabled={!isModified}
            onClick={() => {
              handleBlockSize(
                defaultPlanParams.advanced.sampling
                  .forMonteCarlo.blockSize,
              )
            }}
          >
            Reset to Default
          </button>
        </div>
      </div>
    )

    return isEnabled ? (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        {body}
      </div>
    ) : (
      <button
        className={`${className} params-card w-full text-start relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
        onClick={() => updatePlanParams('setSampling', 'monteCarlo')}
      >
        {body}
      </button>
    )
  },
)

const _HistoricalCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams } = useSimulation()

    const isEnabled = planParams.advanced.sampling.type === 'historical'
    const isModified = useIsHistoricalSequenceCardModifed()
    const body = (
      <div className="">
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className="flex items-start gap-x-2 py-0.5 font-bold text-lg">
          <FontAwesomeIcon
            className="mt-1"
            icon={isEnabled ? faCircleSolid : faCircle}
          />
          <h2 className="">Historical Sequence</h2>
        </div>
        <p className="p-base mt-2">
          Use only actual sequences from the historical data.
        </p>
      </div>
    )

    return isEnabled ? (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        {body}
      </div>
    ) : (
      <button
        className={`${className} params-card w-full text-start relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
        onClick={() => updatePlanParams('setSampling', 'historical')}
      >
        {body}
      </button>
    )
  },
)

export const useIsPlanInputSimulationModifed = () => {
  const isMonteCarloCardModified = useIsMonteCarloCardModifed()
  const isHistoricalSequenceCardModified = useIsHistoricalSequenceCardModifed()
  return isMonteCarloCardModified || isHistoricalSequenceCardModified
}

const useIsMonteCarloCardModifed = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return (
    planParams.advanced.sampling.forMonteCarlo.blockSize !==
    defaultPlanParams.advanced.sampling.forMonteCarlo.blockSize
  )
}
const useIsHistoricalSequenceCardModifed = () => {
  const { planParams } = useSimulation()
  return planParams.advanced.sampling.type === 'historical'
}

export const PlanInputSimulationSummary = React.memo(
  ({ planParams }: { planParams: PlanParams }) => {
    return planParams.advanced.sampling.type === 'monteCarlo' ? (
      <>
        <h2>Monte Carlo Sequence</h2>
        <h2>
          Block Size:{' '}
          {numMonthsStr(
            planParams.advanced.sampling.forMonteCarlo.blockSize,
          )}
        </h2>
      </>
    ) : (
      <h2>Historical Sequence</h2>
    )
  },
)
