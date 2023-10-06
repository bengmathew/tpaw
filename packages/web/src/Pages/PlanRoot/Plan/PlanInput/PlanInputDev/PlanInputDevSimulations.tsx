import { defaultNonPlanParams, historicalReturns } from '@tpaw/common'
import clsx from 'clsx'
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
          <h2> Annual Return Stats:</h2>
          <_AnnualReturnStatsTable className="mt-2" />
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
  const { tpawResult, planParamsProcessed } = useSimulation()
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Num of simulations: {nonPlanParams.numOfSimulationForMonteCarloSampling}
      </h2>
      <h2>Time to run: {_timeToRun(tpawResult)}</h2>
      <h2> Annual Return Stats:</h2>
      <_AnnualReturnStatsTable className="" />
    </>
  )
})

const _timeToRun = (tpawResult: SimulationInfo['tpawResult']) =>
  `${Math.round(tpawResult.perf.main[6][1])}ms (${tpawResult.perf.main[6][0]})`

const _AnnualReturnStatsTable = React.memo(
  ({ className }: { className?: string }) => {
    const { tpawResult, planParamsProcessed } = useSimulation()

    const forData = (stats: {
      ofBase: {
        mean: number
      }
      ofLog: {
        variance: number | null
      }
    }) => (
      <>
        <h2 className="text-right font-mono">
          {formatPercentage(5)(stats.ofBase.mean)}
        </h2>
        <h2 className="text-right font-mono">
          {stats.ofLog.variance?.toFixed(5) ?? '-'}
        </h2>
        <h2 className="text-right font-mono">
          {stats.ofLog.variance
            ? Math.sqrt(stats.ofLog.variance).toFixed(5)
            : '-'}
        </h2>
      </>
    )
    return (
      <div className={clsx(className)}>
        <div
          className="inline-grid gap-x-6 border border-gray-300 rounded-lg p-2"
          style={{ grid: 'auto/auto auto auto auto' }}
        >
          <h2></h2>
          <h2 className="text-center">Mean</h2>
          <div className="text-center">
            Variance <h2 className="text-xs -mt-1">(of Log)</h2>
          </div>
          <div className="text-center">
            SD <h2 className="text-xs -mt-1">(of Log)</h2>
          </div>
          <h2></h2>
          <h2 className="col-span-3 border-t border-gray-300 my-2"></h2>
          <h2>Stocks - Target</h2>
          {forData({
            ofBase: {
              mean: tpawResult.params.expectedReturnsForPlanning.annual.stocks,
            },
            ofLog:
              planParamsProcessed.historicalReturnsAdjusted.monthly.annualStats
                .estimatedSampledStats.stocks.ofLog,
          })}
          <h2>Stocks - Sampled</h2>
          {forData(tpawResult.annualStatsForSampledReturns.stocks)}
          <h2>Stocks - Historical - Adj </h2>
          {forData(
            planParamsProcessed.historicalReturnsAdjusted.monthly.annualStats
              .direct.stocks,
          )}
          <h2>Stocks - Historical - Raw</h2>
          {forData(historicalReturns.monthly.annualStats.stocks)}
          <h2 className="col-span-4 border-t border-gray-300 my-2"></h2>
          <h2>Bonds - Target</h2>
          {forData({
            ofBase: {
              mean: planParamsProcessed.expectedReturnsForPlanning.annual.bonds,
            },
            ofLog: {
              variance: null,
            },
          })}
          <h2>Bonds - Sampled</h2>
          {forData(tpawResult.annualStatsForSampledReturns.bonds)}
          <h2>Bonds - Historical - Adj </h2>
          {forData(
            planParamsProcessed.historicalReturnsAdjusted.monthly.annualStats
              .direct.bonds,
          )}
          <h2>Bonds - Historical - Raw</h2>
          {forData(historicalReturns.monthly.annualStats.bonds)}
        </div>
      </div>
    )
  },
)
