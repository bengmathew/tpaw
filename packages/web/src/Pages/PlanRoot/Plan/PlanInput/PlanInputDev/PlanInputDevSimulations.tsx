import { faDice, faEllipsis } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { errorToast } from '../../../../../Utils/CustomToasts'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { AmountInput } from '../../../../Common/Inputs/AmountInput'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
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
import { useIsPlanInputDevSimulationsModified } from './UseIsPlanInputDevSimulationsModified'

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
    const {
      simulationResult,
      randomSeed,
      reRun,
      simulationResultIsCurrent,
      updatePlanParams,
      planParams,
      defaultPlanParams,
    } = useSimulation()
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const defaultNonPlanParams = useMemo(
      () => getDefaultNonPlanParams(Date.now()),
      [],
    )

    const isModified = useIsPlanInputDevSimulationsModified()

    const [randomSeedInput, setRandomSeedInput] = useState(`${randomSeed}`)
    useEffect(() => {
      setRandomSeedInput(`${randomSeed}`)
    }, [randomSeed])

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Num of Simulations</h2>
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

        <div className="mt-2 flex items-center gap-x-2">
          <h2 className="">Random Seed:</h2>
          <input
            type="number"
            className="text-input"
            value={randomSeedInput}
            onChange={(x) => setRandomSeedInput(x.target.value)}
          />
          <button
            className={clsx(' btn2-xs btn2-dark w-[50px] disabled:lighten-2')}
            disabled={randomSeedInput === randomSeed.toString()}
            onClick={() => {
              let seed = parseInt(randomSeedInput)
              if (isNaN(seed)) {
                errorToast('Not a valid seed')
                return
              }
              reRun(seed)
            }}
          >
            Set
          </button>
          <button
            className=" btn2-xs btn2-dark  w-[50px]"
            onClick={() => reRun('random')}
          >
            <FontAwesomeIcon icon={faDice} />
          </button>
          <button
            className=" disabled:hidden underline text-sm"
            disabled={randomSeed === DEFAULT_MONTE_CARLO_SIMULATION_SEED}
            onClick={() => reRun(DEFAULT_MONTE_CARLO_SIMULATION_SEED)}
          >
            {/* <FontAwesomeIcon icon={faClose} /> */}
            reset
          </button>
        </div>
        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Time to Run:</h2>
          <h2 className="ml-2">{_timeToRun(simulationResult)}</h2>
          <FontAwesomeIcon
            className={clsx('fa-flip ', simulationResultIsCurrent && 'hidden')}
            icon={faEllipsis}
          />
        </div>
        <div className="mt-2 flex items-center gap-x-2">
          <h2 className="">Stagger Run Starts: </h2>
          <ToggleSwitch
            checked={
              planParams.advanced.sampling.forMonteCarlo.staggerRunStarts
            }
            setChecked={(value) => {
              updatePlanParams('setMonteCarloStaggerRunStarts', value)
            }}
          />
        </div>
        <div className="mt-2 flex items-center gap-x-2">
          <h2 className="">Override Historical Returns To Fixed: </h2>
          <ToggleSwitch
            checked={
              planParams.advanced.historicalMonthlyLogReturnsAdjustment
                .overrideToFixedForTesting
            }
            setChecked={(value) => {
              updatePlanParams(
                'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting',
                value,
              )
            }}
          />
        </div>
        <div className="mt-2">
          <h2> Annual Return Stats:</h2>
          <_AnnualReturnStatsTable className="mt-2" />
        </div>

        <button
          className="mt-4 underline disabled:lighten-2 block"
          onClick={() => {
            {
              const clone = _.cloneDeep(nonPlanParams)
              clone.numOfSimulationForMonteCarloSampling =
                defaultNonPlanParams.numOfSimulationForMonteCarloSampling
              setNonPlanParams(clone)
            }
            reRun(DEFAULT_MONTE_CARLO_SIMULATION_SEED)
            updatePlanParams(
              'setMonteCarloStaggerRunStarts',
              defaultPlanParams.advanced.sampling.forMonteCarlo
                .staggerRunStarts,
            )
          }}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const PlanInputDevSimulationsSummary = React.memo(() => {
  const { simulationResult, randomSeed, planParams } = useSimulation()
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Num of Simulations: {nonPlanParams.numOfSimulationForMonteCarloSampling}
      </h2>
      <h2>Random Seed: {randomSeed}</h2>
      <h2>Time to Run: {_timeToRun(simulationResult)}</h2>
      <h2>
        Stagger Run Starts:{' '}
        {planParams.advanced.sampling.forMonteCarlo.staggerRunStarts
          ? 'yes'
          : 'no'}
      </h2>
      <h2>
        Override Historical Returns To Fixed:{' '}
        {planParams.advanced.historicalMonthlyLogReturnsAdjustment
          .overrideToFixedForTesting
          ? 'yes'
          : 'no'}
      </h2>
      <h2> Annual Return Stats:</h2>
      <_AnnualReturnStatsTable className="" />
    </>
  )
})

const _timeToRun = (simulationResult: SimulationInfo['simulationResult']) =>
  `${Math.round(simulationResult.perf.main[6][1])}ms (${
    simulationResult.perf.main[6][0]
  })`

const _AnnualReturnStatsTable = React.memo(
  ({ className }: { className?: string }) => {
    const { simulationResult, planParamsProcessed } = useSimulation()

    const forData = (stats: {
      ofBase: {
        mean: number
      }
      ofLog: {
        variance: number
        standardDeviation?: number
      }
    }) => (
      <>
        <h2 className="text-right font-mono">
          {formatPercentage(5)(stats.ofBase.mean)}
        </h2>
        <h2 className="text-right font-mono">
          {stats.ofLog.variance.toFixed(5)}
        </h2>
        <h2 className="text-right font-mono">
          {(
            stats.ofLog.standardDeviation ?? Math.sqrt(stats.ofLog.variance)
          ).toFixed(5)}
        </h2>
      </>
    )

    const targetData = {
      stocks: {
        ofBase: {
          mean: simulationResult.args.planParamsProcessed
            .expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.stocks
            .value,
        },
        ofLog: {
          variance:
            planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
              .empiricalAnnualLogVariance,
        },
      },
      bonds: {
        ofBase: {
          mean: simulationResult.args.planParamsProcessed
            .expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.bonds
            .value,
        },
        ofLog: {
          variance:
            planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.stats
              .empiricalAnnualLogVariance,
        },
      },
    }
    const delta = (
      a: (typeof targetData)['stocks'],
      b: (typeof targetData)['stocks'],
    ) => ({
      ofBase: {
        mean: a.ofBase.mean - b.ofBase.mean,
      },
      ofLog: {
        variance: a.ofLog.variance - b.ofLog.variance,
        standardDeviation:
          Math.sqrt(a.ofLog.variance) - Math.sqrt(b.ofLog.variance),
      },
    })

    return (
      <div className={clsx(className)}>
        <div
          className="inline-grid gap-x-6 border border-gray-300 rounded-lg p-2"
          style={{ grid: 'auto/auto auto auto auto' }}
        >
          <h2></h2>
          <h2 className="text-center">Mean</h2>
          <div className="text-center">
            <h2 className="">Variance</h2>{' '}
            <h2 className="text-xs lighten -mt-1">(of Log)</h2>
          </div>
          <div className="text-center">
            <h2 className="">SD</h2>
            <h2 className="text-xs lighten -mt-1">(of Log)</h2>
          </div>
          <h2></h2>
          <h2 className="col-span-4 border-t border-gray-300 my-2"></h2>
          <h2>Stocks - Target</h2>
          {forData(targetData.stocks)}
          <h2>Stocks - Sampled</h2>
          {forData(simulationResult.annualStatsForSampledReturns.stocks)}
          <h2>Stocks - Sampled 𝚫</h2>
          {forData(
            delta(
              simulationResult.annualStatsForSampledReturns.stocks,
              targetData.stocks,
            ),
          )}
          <h2>Stocks - Historical - Adj </h2>
          {forData({
            ofBase:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
                .annualized.nonLog,
            ofLog:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
                .annualized.log,
          })}
          <h2>Stocks - Historical - Raw</h2>
          {forData({
            ofBase:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
                .unadjustedAnnualized.nonLog,
            ofLog:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
                .unadjustedAnnualized.log,
          })}
          <h2 className="col-span-4 border-t border-gray-300 my-2"></h2>
          <h2>Bonds - Target</h2>
          {forData(targetData.bonds)}
          <h2>Bonds - Sampled</h2>
          {forData(simulationResult.annualStatsForSampledReturns.bonds)}
          <h2>Bonds - Sampled - 𝚫</h2>
          {forData(
            delta(
              simulationResult.annualStatsForSampledReturns.bonds,
              targetData.bonds,
            ),
          )}
          <h2>Bonds - Historical - Adj </h2>
          {forData({
            ofBase:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.stats
                .annualized.nonLog,
            ofLog:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.stats
                .annualized.log,
          })}
          <h2>Bonds - Historical - Raw</h2>
          {forData({
            ofBase:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.stats
                .unadjustedAnnualized.nonLog,
            ofLog:
              planParamsProcessed.historicalMonthlyReturnsAdjusted.bonds.stats
                .unadjustedAnnualized.log,
          })}
        </div>
      </div>
    )
  },
)
