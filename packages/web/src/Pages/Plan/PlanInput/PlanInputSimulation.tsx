import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MAX_AGE_IN_MONTHS } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { paddingCSSStyle, paddingCSSStyleHorz } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { NumMonthsInput } from '../../Common/Inputs/NumMonthsInput'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputSimulation = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { setPlanParams, defaultParams } = useSimulation()

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
            onClick={() => {
              setPlanParams((plan) => {
                const clone = _.cloneDeep(plan)
                clone.advanced.sampling = defaultParams.plan.advanced.sampling
                clone.advanced.monteCarloSampling.blockSize =
                  defaultParams.plan.advanced.monteCarloSampling.blockSize
                return clone
              })
            }}
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
    const { params, setPlanParams, defaultParams } = useSimulation()
    const defaultNumBlocks =
      defaultParams.plan.advanced.monteCarloSampling.blockSize
    const isEnabled = params.plan.advanced.sampling === 'monteCarlo'
    const handleBlockSize = (x: number) => {
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        clone.advanced.monteCarloSampling.blockSize = x
        return clone
      })
    }
    const body = (
      <div className="">
        <PlanInputModifiedBadge
          show={
            isEnabled &&
            params.plan.advanced.monteCarloSampling.blockSize !==
              defaultNumBlocks
          }
          mainPage={false}
        />
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
            value={params.plan.advanced.monteCarloSampling.blockSize}
            onChange={handleBlockSize}
            rangeAsMFN={{ start: 1, end: MAX_AGE_IN_MONTHS }}
            disabled={!isEnabled}
          />
          <button
            className="mt-4 underline disabled:lighten-2"
            disabled={
              params.plan.advanced.monteCarloSampling.blockSize ===
              defaultNumBlocks
            }
            onClick={() => {
              handleBlockSize(defaultNumBlocks)
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
        onClick={() => {
          setPlanParams((plan) => {
            const clone = _.cloneDeep(plan)
            clone.advanced.sampling = 'monteCarlo'
            return clone
          })
        }}
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
    const { params, setPlanParams } = useSimulation()

    const isEnabled = params.plan.advanced.sampling === 'historical'
    const body = (
      <div className="">
        <PlanInputModifiedBadge show={isEnabled} mainPage={false} />
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
        onClick={() => {
          setPlanParams((plan) => {
            const clone = _.cloneDeep(plan)
            clone.advanced.sampling = 'historical'
            return clone
          })
        }}
      >
        {body}
      </button>
    )
  },
)
