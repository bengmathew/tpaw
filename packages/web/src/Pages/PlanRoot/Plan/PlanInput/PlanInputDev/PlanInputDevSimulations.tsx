import { faDice, faEllipsis } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  assert,
  partialDefaultDatelessPlanParams,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { errorToast } from '../../../../../Utils/CustomToasts'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { Record } from '../../../../../Utils/Record'
import { AmountInput } from '../../../../Common/Inputs/AmountInput'
import { SwitchAsToggle } from '../../../../Common/Inputs/SwitchAsToggle'
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
      planParamsNorm,
    } = useSimulation()
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const defaultNonPlanParams = useMemo(
      () => getDefaultNonPlanParams(Date.now()),
      [],
    )
    assert(partialDefaultDatelessPlanParams.advanced.sampling.type === 'monteCarlo')

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
          <SwitchAsToggle
            checked={
              planParamsNorm.advanced.sampling.type === 'monteCarlo'
                ? planParamsNorm.advanced.sampling.data.staggerRunStarts
                : planParamsNorm.advanced.sampling.defaultData.monteCarlo
                    ?.staggerRunStarts ??
                  partialDefaultDatelessPlanParams.advanced.sampling.data
                    .staggerRunStarts
            }
            disabled={planParamsNorm.advanced.sampling.type !== 'monteCarlo'}
            setChecked={(value) => {
              updatePlanParams('setMonteCarloStaggerRunStarts', value)
            }}
          />
        </div>
        <div className="mt-2 flex items-center gap-x-2">
          <h2 className="">Override Historical Returns To Fixed: </h2>
          <SwitchAsToggle
            checked={
              planParamsNorm.advanced.historicalReturnsAdjustment
                .overrideToFixedForTesting.type !== 'none'
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
            assert(
              partialDefaultDatelessPlanParams.advanced.sampling.type === 'monteCarlo',
            )
            if (
              planParamsNorm.advanced.sampling.type === 'monteCarlo' &&
              planParamsNorm.advanced.sampling.data.staggerRunStarts !==
                partialDefaultDatelessPlanParams.advanced.sampling.data.staggerRunStarts
            )
              updatePlanParams(
                'setMonteCarloStaggerRunStarts',
                partialDefaultDatelessPlanParams.advanced.sampling.data
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
  const { simulationResult, randomSeed, planParamsNorm } = useSimulation()
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
        {planParamsNorm.advanced.sampling.type !== 'monteCarlo'
          ? 'N/A'
          : planParamsNorm.advanced.sampling.data.staggerRunStarts
            ? 'yes'
            : 'no'}
      </h2>
      <h2>
        Override Historical Returns To Fixed:{' '}
        {planParamsNorm.advanced.historicalReturnsAdjustment
          .overrideToFixedForTesting.type !== 'none'
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
          <_AnnualReturnStatsTableSection type="stocks" />
          <div className=" col-span-4 my-1.5 border-b border-gray-300"></div>
          <_AnnualReturnStatsTableSection type="bonds" />
        </div>
      </div>
    )
  },
)

const _AnnualReturnStatsTableSection = React.memo(
  ({ type }: { type: 'stocks' | 'bonds' }) => {
    const { simulationResult } = useSimulation()
    const { planParamsProcessed } = simulationResult.args
    const { returnsStatsForPlanning, historicalReturnsAdjusted } =
      planParamsProcessed

    const addSd = (mean: number, varianceOfLog: number) => ({
      mean,
      varianceOfLog,
      sdOfLog: Math.sqrt(varianceOfLog),
    })

    const targetForPlanning = addSd(
      returnsStatsForPlanning[type].empiricalAnnualNonLogExpectedReturnInfo
        .value,
      returnsStatsForPlanning[type].empiricalAnnualLogVariance,
    )
    const targetForSimulation = addSd(
      historicalReturnsAdjusted[type].args
        .empiricalAnnualNonLogExpectedReturnInfo.value,
      historicalReturnsAdjusted[type].args.empiricalAnnualLogVariance,
    )
    const sampled = addSd(
      simulationResult.annualStatsForSampledReturns[type].ofBase.mean,
      simulationResult.annualStatsForSampledReturns[type].ofLog.variance,
    )

    const adjusted = addSd(
      historicalReturnsAdjusted[type].stats.annualized.nonLog.mean,
      historicalReturnsAdjusted[type].stats.annualized.log.variance,
    )

    const unadjusted = addSd(
      historicalReturnsAdjusted[type].srcAnnualizedStats.nonLog.mean,
      historicalReturnsAdjusted[type].srcAnnualizedStats.log.variance,
    )

    const label = _.upperFirst(type)
    return (
      <>
        <_AnnualReturnStatsTableRow
          label={`${label} - For Planning`}
          {...targetForPlanning}
        />
        <_AnnualReturnStatsTableRow
          label={`${label} - For Simulation`}
          {...targetForSimulation}
        />
        <_AnnualReturnStatsTableRow label={`${label} - Sampled`} {...sampled} />
        <_AnnualReturnStatsTableRow
          label={`${label} - Sampled 𝚫`}
          {...Record.mapValues(
            sampled,
            (sampled, key) => sampled - targetForSimulation[key],
          )}
        />
        <_AnnualReturnStatsTableRow
          label={`${label} - Historical -Adj`}
          {...adjusted}
        />
        <_AnnualReturnStatsTableRow
          label={`${label} - Historical -Raw`}
          {...unadjusted}
        />
      </>
    )
  },
)

const _AnnualReturnStatsTableRow = React.memo(
  ({
    mean,
    varianceOfLog,
    sdOfLog,
    label,
  }: {
    mean: number
    varianceOfLog: number
    sdOfLog: number
    label: string
  }) => {
    return (
      <>
        <h2>{label}</h2>
        <h2 className="text-right font-mono">{formatPercentage(5)(mean)}</h2>
        <h2 className="text-right font-mono">{varianceOfLog.toFixed(5)}</h2>
        <h2 className="text-right font-mono">{sdOfLog.toFixed(5)}</h2>
      </>
    )
  },
)
