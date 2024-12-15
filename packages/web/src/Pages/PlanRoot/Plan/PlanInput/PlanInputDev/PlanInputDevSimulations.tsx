import { faDice, faEllipsis } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  PlanParams,
  assert,
  block,
  getDefaultNonPlanParams,
  partialDefaultDatelessPlanParams,
} from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { errorToast } from '../../../../../Utils/CustomToasts'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { AmountInput } from '../../../../Common/Inputs/AmountInput'
import { SwitchAsToggle } from '../../../../Common/Inputs/SwitchAsToggle'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {
  useIsPlanInputDevOverrideHistoricalReturnsToFixedForTestingModified,
  useIsPlanInputDevSimulationsMainCardModified,
  useIsPlanInputDevSimulationsModified,
} from './UseIsPlanInputDevSimulationsModified'
import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { SimulationResult2 } from '../../../../../Simulator/UseSimulator'

export const PlanInputDevSimulations = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_SimulationsCard className="mt-10" props={props} />
          <_OverrideHistoricalReturnsToFixedForTestingCard
            className="mt-10"
            props={props}
          />
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
    const { randomSeed, reRun, updatePlanParams, planParamsNormInstant } =
      useSimulationInfo()
    const { simulationResult, simulationIsRunningInfo } =
      useSimulationResultInfo()
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const defaultNonPlanParams = useMemo(
      () => getDefaultNonPlanParams(Date.now()),
      [],
    )
    assert(
      partialDefaultDatelessPlanParams.advanced.sampling.type === 'monteCarlo',
    )

    const isModified = useIsPlanInputDevSimulationsMainCardModified()

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
          <h2 className="">Override Num of Simulations:</h2>
          <SwitchAsToggle
            checked={
              nonPlanParams.numOfSimulationForMonteCarloSampling !== 'default'
            }
            setChecked={(checked) => {
              const clone = _.cloneDeep(nonPlanParams)
              clone.numOfSimulationForMonteCarloSampling = checked
                ? simulationResult.numOfSimulationForMonteCarloSamplingOfResult
                : 'default'
              setNonPlanParams(clone)
            }}
          />

          {nonPlanParams.numOfSimulationForMonteCarloSampling !== 'default' && (
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
          )}
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
          <h2 className="">
            Status:{' '}
            {simulationIsRunningInfo.isRunning ? 'Running...' : 'Completed'}
          </h2>
        </div>
        <div className="mt-2 flex items-center gap-x-2">
          <h2 className="">Stagger Run Starts: </h2>
          <SwitchAsToggle
            checked={
              planParamsNormInstant.advanced.sampling.type === 'monteCarlo'
                ? planParamsNormInstant.advanced.sampling.data.staggerRunStarts
                : (planParamsNormInstant.advanced.sampling.defaultData
                    .monteCarlo?.staggerRunStarts ??
                  partialDefaultDatelessPlanParams.advanced.sampling.data
                    .staggerRunStarts)
            }
            disabled={
              planParamsNormInstant.advanced.sampling.type !== 'monteCarlo'
            }
            setChecked={(value) => {
              updatePlanParams('setMonteCarloStaggerRunStarts', value)
            }}
          />
        </div>
        <_PerformanceTable
          className="mt-2"
          stats={simulationResult.performance}
        />

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
              partialDefaultDatelessPlanParams.advanced.sampling.type ===
                'monteCarlo',
            )
            if (
              planParamsNormInstant.advanced.sampling.type === 'monteCarlo' &&
              planParamsNormInstant.advanced.sampling.data.staggerRunStarts !==
                partialDefaultDatelessPlanParams.advanced.sampling.data
                  .staggerRunStarts
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

const _PerformanceTable = ({
  className = '',
  stats,
}: {
  className?: string
  stats: SimulationResult2['performance']
}) => {
  return (
    <div className={clsx(className)}>
      <h2 className="">Performance:</h2>
      <div
        className="ml-4 grid gap-x-4 items-center"
        style={{ grid: 'auto/auto 1fr' }}
      >
        <h2 className="">Client:</h2>
        <h2 className="ml-2">{Math.round(stats.client_time_in_ms)}ms</h2>
        <h2 className="">Server Total:</h2>
        <h2 className="ml-2">{stats.server_time_in_ms.totalInMs}ms</h2>
        <h2 className="">Server PBE:</h2>
        <h2 className="ml-2">
          {stats.server_time_in_ms.portfolioBalanceEstimationInMs}ms
        </h2>
        <h2 className="">Server Simulation:</h2>
        <h2 className="ml-2">{stats.server_time_in_ms.simulationInMs}ms</h2>
        <h2 className="">Upload size (zip):</h2>
        <h2 className="ml-2">
          {(stats.compressed_upload_payload_in_bytes / 1000).toFixed(1)}
          KB
        </h2>
      </div>
    </div>
  )
}

const _OverrideHistoricalReturnsToFixedForTestingCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const isModified =
      useIsPlanInputDevOverrideHistoricalReturnsToFixedForTestingModified()
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const { planParamsProcessed } = useSimulationResultInfo().simulationResult
    const curr =
      planParamsNormInstant.advanced.historicalReturnsAdjustment
        .overrideToFixedForTesting

    const handleUpdate = (
      value: PlanParams['advanced']['historicalReturnsAdjustment']['overrideToFixedForTesting'],
    ) => {
      updatePlanParams(
        'setHistoricalMonthlyLogReturnsAdjustmentOverrideToFixedForTesting2',
        value,
      )
    }
    const [manualStocks, setManualStocks] = useState(
      planParamsNormInstant.advanced.historicalReturnsAdjustment
        .overrideToFixedForTesting.type === 'manual'
        ? planParamsNormInstant.advanced.historicalReturnsAdjustment
            .overrideToFixedForTesting.stocks
        : null,
    )
    const [manualBonds, setManualBonds] = useState(
      planParamsNormInstant.advanced.historicalReturnsAdjustment
        .overrideToFixedForTesting.type === 'manual'
        ? planParamsNormInstant.advanced.historicalReturnsAdjustment
            .overrideToFixedForTesting.bonds
        : null,
    )
    const expectedStocks =
      planParamsProcessed.returnsStatsForPlanning.stocks
        .empiricalAnnualNonLogExpectedReturn
    const expectedBonds =
      planParamsProcessed.returnsStatsForPlanning.bonds
        .empiricalAnnualNonLogExpectedReturn

    const updateManual = () => {
      const stocks = manualStocks ?? expectedStocks
      const bonds = manualBonds ?? expectedBonds
      handleUpdate({ type: 'manual', stocks, bonds })
      setManualStocks(stocks)
      setManualBonds(bonds)
    }

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <h2 className="font-bold text-lg ">
          Override Historical Returns To Fixed
        </h2>
        <div className="mt-2">
          <button
            className="flex items-center gap-x-2 py-2"
            onClick={() => handleUpdate({ type: 'none' })}
          >
            <FontAwesomeIcon
              icon={curr.type === 'none' ? faCircleSelected : faCircleRegular}
            />
            {"Don't"} Override
          </button>
          <button
            className="flex items-start  gap-x-2 py-2"
            onClick={() =>
              handleUpdate({ type: 'useExpectedReturnsForPlanning' })
            }
          >
            <FontAwesomeIcon
              className="mt-1"
              icon={
                curr.type === 'useExpectedReturnsForPlanning'
                  ? faCircleSelected
                  : faCircleRegular
              }
            />
            Override to Expected Returns
          </button>
          <button
            className="flex items-start gap-x-2 py-2"
            onClick={updateManual}
          >
            <FontAwesomeIcon
              className="mt-1"
              icon={curr.type === 'manual' ? faCircleSelected : faCircleRegular}
            />
            Manual
          </button>
          {curr.type === 'manual' && (
            <div
              className="pl-6 inline-grid items-center gap-x-2 gap-y-2"
              style={{ grid: 'auto/auto auto' }}
            >
              <h2 className="">Stocks:</h2>
              <input
                className=" text-input w-[150px]"
                type="number"
                value={manualStocks ?? expectedStocks}
                onChange={(x) => setManualStocks(x.target.valueAsNumber)}
                onBlur={updateManual}
              />
              <h2 className="">Bonds:</h2>
              <input
                className=" text-input w-[150px]"
                type="number"
                value={manualBonds ?? expectedBonds}
                onChange={(x) => setManualBonds(x.target.valueAsNumber)}
                onBlur={updateManual}
              />
            </div>
          )}
        </div>

        <button
          className="mt-4 underline disabled:lighten-2 block"
          onClick={() => {
            handleUpdate(
              partialDefaultDatelessPlanParams.advanced
                .historicalReturnsAdjustment.overrideToFixedForTesting,
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
  const { randomSeed, planParamsNormInstant } = useSimulationInfo()
  const { simulationResult } = useSimulationResultInfo()
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Num of Simulations: {nonPlanParams.numOfSimulationForMonteCarloSampling}
      </h2>
      <h2>Random Seed: {randomSeed}</h2>
      <h2>
        Stagger Run Starts:{' '}
        {planParamsNormInstant.advanced.sampling.type !== 'monteCarlo'
          ? 'N/A'
          : planParamsNormInstant.advanced.sampling.data.staggerRunStarts
            ? 'yes'
            : 'no'}
      </h2>
      <_PerformanceTable stats={simulationResult.performance} />
      <h2>Override Historical Returns To Fixed: </h2>
      <h2 className="pl-4">
        {block(() => {
          const value =
            planParamsNormInstant.advanced.historicalReturnsAdjustment
              .overrideToFixedForTesting
          switch (value.type) {
            case 'none':
              return 'No Override'
            case 'useExpectedReturnsForPlanning':
              return 'Expected Returns'
            case 'manual':
              return `Manual: (stocks: ${formatPercentage(1)(value.stocks)}, bonds: ${formatPercentage(1)(value.bonds)})`
          }
        })}
      </h2>
    </>
  )
})

// TODO: After duration matching. What are the details to show here?
// const _AnnualReturnStatsTable = React.memo(
//   ({ className }: { className?: string }) => {
//     return (
//       <div className={clsx(className)}>
//         <div
//           className="inline-grid gap-x-6 border border-gray-300 rounded-lg p-2"
//           style={{ grid: 'auto/auto auto auto auto' }}
//         >
//           <h2></h2>
//           <h2 className="text-center">Mean</h2>
//           <div className="text-center">
//             <h2 className="">Variance</h2>{' '}
//             <h2 className="text-xs lighten -mt-1">(of Log)</h2>
//           </div>
//           <div className="text-center">
//             <h2 className="">SD</h2>
//             <h2 className="text-xs lighten -mt-1">(of Log)</h2>
//           </div>
//           <h2></h2>
//           <h2 className="col-span-4 border-t border-gray-300 my-2"></h2>
//           <_AnnualReturnStatsTableSection type="stocks" />
//           <div className=" col-span-4 my-1.5 border-b border-gray-300"></div>
//           <_AnnualReturnStatsTableSection type="bonds" />
//         </div>
//       </div>
//     )
//   },
// )

// const _AnnualReturnStatsTableSection = React.memo(
//   ({ type }: { type: 'stocks' | 'bonds' }) => {
//     const { simulationResult } = useSimulationInfo()
//     const { planParamsProcessed } = simulationResult
//     const { returnsStatsForPlanning, historicalReturnsAdjusted } =
//       planParamsProcessed

//     const addSd = (mean: number, varianceOfLog: number) => ({
//       mean,
//       varianceOfLog,
//       sdOfLog: Math.sqrt(varianceOfLog),
//     })

//     const targetForPlanning = addSd(
//       returnsStatsForPlanning[type].empiricalAnnualNonLogExpectedReturnInfo
//         .value,
//       returnsStatsForPlanning[type].empiricalAnnualLogVariance,
//     )
//     const targetForSimulation = addSd(
//       historicalReturnsAdjusted[type].args
//         .empiricalAnnualNonLogExpectedReturnInfo.value,
//       historicalReturnsAdjusted[type].args.empiricalAnnualLogVariance,
//     )
//     const sampled = addSd(
//       simulationResult.annualStatsForSampledReturns[type].ofBase.mean,
//       simulationResult.annualStatsForSampledReturns[type].ofLog.variance,
//     )

//     const adjusted = addSd(
//       historicalReturnsAdjusted[type].stats.annualized.nonLog.mean,
//       historicalReturnsAdjusted[type].stats.annualized.log.variance,
//     )

//     const unadjusted = addSd(
//       historicalReturnsAdjusted[type].srcAnnualizedStats.nonLog.mean,
//       historicalReturnsAdjusted[type].srcAnnualizedStats.log.variance,
//     )

//     const label = _.upperFirst(type)
//     return (
//       <>
//         <_AnnualReturnStatsTableRow
//           label={`${label} - For Planning`}
//           {...targetForPlanning}
//         />
//         <_AnnualReturnStatsTableRow
//           label={`${label} - For Simulation`}
//           {...targetForSimulation}
//         />
//         <_AnnualReturnStatsTableRow label={`${label} - Sampled`} {...sampled} />
//         <_AnnualReturnStatsTableRow
//           label={`${label} - Sampled 𝚫`}
//           {...Record.mapValues(
//             sampled,
//             (sampled, key) => sampled - targetForSimulation[key],
//           )}
//         />
//         <_AnnualReturnStatsTableRow
//           label={`${label} - Historical -Adj`}
//           {...adjusted}
//         />
//         <_AnnualReturnStatsTableRow
//           label={`${label} - Historical -Raw`}
//           {...unadjusted}
//         />
//       </>
//     )
//   },
// )

// const _AnnualReturnStatsTableRow = React.memo(
//   ({
//     mean,
//     varianceOfLog,
//     sdOfLog,
//     label,
//   }: {
//     mean: number
//     varianceOfLog: number
//     sdOfLog: number
//     label: string
//   }) => {
//     return (
//       <>
//         <h2>{label}</h2>
//         <h2 className="text-right font-mono">{formatPercentage(5)(mean)}</h2>
//         <h2 className="text-right font-mono">{varianceOfLog.toFixed(5)}</h2>
//         <h2 className="text-right font-mono">{sdOfLog.toFixed(5)}</h2>
//       </>
//     )
//   },
// )
