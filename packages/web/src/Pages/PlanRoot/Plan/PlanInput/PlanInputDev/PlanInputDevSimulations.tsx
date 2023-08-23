import { defaultNonPlanParams } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { getTPAWRunInWorkerSingleton } from '../../../../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { AmountInput } from '../../../../Common/Inputs/AmountInput'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfo,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevSimulations = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_SimulationsCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)

const _SimulationsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { tpawResult, reRun } = useSimulation()
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()

    const isModified = useIsPlanInputDevSimulationsModified()

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Number of simulations</h2>
          <AmountInput
            className="text-input"
            value={nonPlanParams.numOfSimulationForMonteCarloSampling}
            onChange={(numOfSimulations) => {
              const clone = _.cloneDeep(nonPlanParams)
              clone.numOfSimulationForMonteCarloSampling = numOfSimulations
              setNonPlanParams(clone)
            }}
            decimals={0}
            modalLabel="Number of Simulations"
          />
        </div>

        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Time To Run:</h2>
          <h2 className="ml-2">{_timeToRun(tpawResult)}</h2>
        </div>
        <div className="mt-2">
          <h2 className="mb-1">Average Sampled Annual Returns:</h2>
          <div className="ml-4 flex gap-x-4 items-center">
            <h2 className="">Stocks</h2>
            <h2 className="ml-2">
              {formatPercentage(5)(tpawResult.averageAnnualReturns.stocks)}
            </h2>
          </div>
          <div className="ml-4 flex gap-x-4 items-center">
            <h2 className="">Bonds</h2>
            <h2 className="ml-2">
              {formatPercentage(5)(tpawResult.averageAnnualReturns.bonds)}
            </h2>
          </div>
        </div>

        <button
          className="underline pt-4"
          onClick={() => {
            void getTPAWRunInWorkerSingleton().clearMemoizedRandom()
            reRun()
          }}
        >
          Reset random draws
        </button>
        <button
          className="mt-6 underline disabled:lighten-2 block"
          onClick={() => {
            const clone = _.cloneDeep(nonPlanParams)
            clone.numOfSimulationForMonteCarloSampling =
              defaultNonPlanParams.numOfSimulationForMonteCarloSampling
            setNonPlanParams(clone)
          }}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const useIsPlanInputDevSimulationsModified = () => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    nonPlanParams.numOfSimulationForMonteCarloSampling !==
    defaultNonPlanParams.numOfSimulationForMonteCarloSampling
  )
}

export const PlanInputDevSimulationsSummary = React.memo(() => {
  const { tpawResult } = useSimulation()
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Num of simulations: {nonPlanParams.numOfSimulationForMonteCarloSampling}
      </h2>
      <h2>Time to run: {_timeToRun(tpawResult)}</h2>
      <h2>Average Sampled Annual Returns:</h2>
      <h2 className="ml-4">
        Stocks: {formatPercentage(5)(tpawResult.averageAnnualReturns.stocks)}
      </h2>
      <h2 className="ml-4">
        Bonds: {formatPercentage(5)(tpawResult.averageAnnualReturns.bonds)}
      </h2>
    </>
  )
})

const _timeToRun = (tpawResult: SimulationInfo['tpawResult']) =>
  `${Math.round(tpawResult.perf.main[6][1])}ms (${tpawResult.perf.main[6][0]})`
