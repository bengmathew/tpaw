import {
  CalendarDay,
  PlanParams,
  PlanParamsChangeActionCurrent,
  SomePlanParamsVersion,
  fGet,
  getNYZonedTime,
  getZonedTimeFns,
  noCase,
  nonPlanParamFns,
} from '@tpaw/common'
import cloneJSON from 'fast-json-clone'
import React, {
  useCallback,
  useEffect,
  useInsertionEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { useGlobalSuspenseFallbackContext } from '../../../../pages/_app'
import { appPaths } from '../../../AppPaths'
import { PlanParamsNormalized } from '../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from '../../../Simulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { DailyMarketSeriesSrc } from '../../../Simulator/SimulateOnServer/SimulateOnServer'
import {
  SimulationResult,
  useSimulator,
} from '../../../Simulator/UseSimulator'
import { createContext } from '../../../Utils/CreateContext'
import { infoToast } from '../../../Utils/CustomToasts'
import { UnionToIntersection } from '../../../Utils/UnionToIntersection'
import { useAssertConst } from '../../../Utils/UseAssertConst'
import { useURLParam } from '../../../Utils/UseURLParam'
import { User, useUser } from '../../App/WithUser'
import { Plan } from '../Plan/Plan'
import { PlanFileData, PlanFileDataFns } from '../PlanRootFile/PlanFileData'
import { ServerSyncState } from '../PlanServerImpl/UseServerSyncPlan'
import {
  PlanPrintViewArgs,
  PlanPrintViewSettingsClientSide,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintView/PlanPrintViewArgs'
import { CurrentTimeInfo } from './UseCurrentTime'
import { WorkingPlanInfo } from './UseWorkingPlan'
import { useIANATimezoneName, useNonPlanParams } from './WithNonPlanParams'

export const DEFAULT_MONTE_CARLO_SIMULATION_SEED = 860336713
const PERCENTILES = { low: 5, mid: 50, high: 95 }
type UpdatePlanParamsFromAction<T> = UnionToIntersection<
  T extends PlanParamsChangeActionCurrent
    ? (x: T['type'], v: T['value']) => void
    : never
>
export type UpdatePlanParams =
  UpdatePlanParamsFromAction<PlanParamsChangeActionCurrent>

export type SimulationInfo = {
  planId: string
  planPaths: (typeof appPaths)['plan']
  fastForwardInfo: CurrentTimeInfo['fastForwardInfo']

  planParamsId: string
  // "Instant" to distinguish from planParamsNormInResult which matches the
  // result exactly, while this instantly reflects the changes made by the user.
  planParamsNormInstant: PlanParamsNormalized
  planMigratedFromVersion: SomePlanParamsVersion

  updatePlanParams: UpdatePlanParams
  planParamsHistoryDays: Set<number>

  simulationInfoBySrc:
    | {
        src: 'localMain'
        reset: (planParams: PlanParams) => void
      }
    | {
        src: 'file'
        setSrc: (filename: string | null, data: PlanFileData) => void
        isModified: boolean
        reset: (planParams: PlanParams | null) => void
        plan: {
          filename: string | null
          convertedToFilePlanAtTimestamp: number
        }
        setForceNav: () => void
      }
    | {
        src: 'server'
        plan: User['plans'][0]
        historyStatus: 'fetching' | 'fetched' | 'failed'
        syncState: ServerSyncState
        setRewindTo: (x: 'lastUpdate' | CalendarDay | null) => void
        reload: () => void
      }
    | {
        src: 'link'
        isModified: boolean
        reset: (planParams: PlanParams | null) => void
        setForceNav: () => void
      }

  simulationInfoByMode:
    | {
        mode: 'plan'
        planParamsUndoRedoStack: WorkingPlanInfo['planParamsUndoRedoStack']
        setPlanParamsHeadIndex: (x: number) => void
      }
    | {
        mode: 'history'
        actualCurrentTimestamp: number
      }

  numOfSimulationForMonteCarloSampling: number
  randomSeed: number

  reRun: (seed: 'random' | number) => void
}

export type SimulationInfoForLocalMainSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: 'localMain' }
>
export type SimulationInfoForFileSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: 'file' }
>
export type SimulationInfoForSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: '' }
>
export type SimulationInfoForServerSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: 'server' }
>
export type SimulationInfoForLinkSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: 'link' }
>
export type SimulationInfoForServerSidePrintSrc = Extract<
  SimulationInfo['simulationInfoBySrc'],
  { src: 'serverSidePrint' }
>
export type SimulationInfoForPlanMode = Extract<
  SimulationInfo['simulationInfoByMode'],
  { mode: 'plan' }
>
export type SimulationInfoForHistoryMode = Extract<
  SimulationInfo['simulationInfoByMode'],
  { mode: 'history' }
>

export type SimulationParams = Omit<
  SimulationInfo,
  | 'planParamsId'
  | 'planParamsNormInstant'
  | 'planParamsHistoryDays'
  | 'numOfSimulationForMonteCarloSampling'
  | 'randomSeed'
  | 'reRun'
> & {
  simulationTimestamp: number
  planParamsHistoryPreBase: null | readonly { id: string; params: PlanParams }[]
  planParamsHistoryPostBase: { id: string; params: PlanParams }[]

  pdfReportInfo:
    | {
        isShowing: true
        onSettings: (settings: PlanPrintViewSettingsClientSide) => void
      }
    | {
        isShowing: false
        show: (args: {
          fixed: PlanPrintViewArgs['fixed']
          settings: PlanPrintViewSettingsClientSide
          simulationResult: SimulationResult | null
          updateSettings: (x: PlanPrintViewSettingsControlledClientSide) => void
        }) => void
      }
}

export const getSimulationParamsForPlanMode = (
  planPaths: SimulationInfo['planPaths'],
  currentTimeInfo: CurrentTimeInfo,
  workingPlanInfo: WorkingPlanInfo,
  planParamsHistoryPreBase:
    | null
    | readonly { id: string; params: PlanParams }[],
  planMigratedFromVersion: SomePlanParamsVersion,
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
  pdfReportInfo: SimulationParams['pdfReportInfo'],
): SimulationParams => ({
  planId: workingPlanInfo.workingPlan.planId,
  planPaths,
  planParamsHistoryPreBase,
  planParamsHistoryPostBase: workingPlanInfo.planParamsUndoRedoStack.undos,

  simulationTimestamp: currentTimeInfo.currentTimestamp,
  fastForwardInfo: currentTimeInfo.fastForwardInfo,
  planMigratedFromVersion,

  updatePlanParams: workingPlanInfo.updatePlanParams,
  simulationInfoBySrc,
  simulationInfoByMode: {
    mode: 'plan',
    planParamsUndoRedoStack: workingPlanInfo.planParamsUndoRedoStack,
    setPlanParamsHeadIndex: workingPlanInfo.setPlanParamsHeadIndex,
  },
  pdfReportInfo,
})

export const useSimulationParamsForHistoryMode = (
  // Null means not actually history mode, we can return null.
  rewindInfo: {
    rewindTo: number
    planParamsHistoryPreBase: { id: string; params: PlanParams }[]
  } | null,
  planPaths: SimulationInfo['planPaths'],
  currentTimeInfo: CurrentTimeInfo,
  workingPlanInfo: WorkingPlanInfo,
  planMigratedFromVersion: SomePlanParamsVersion,
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
  pdfReportInfo: SimulationParams['pdfReportInfo'],
): SimulationParams | null => {
  const updatePlanParams = useCallback(
    () => infoToast('Cannot update plan in history mode.'),
    [],
  )

  if (!rewindInfo) return null

  return {
    planId: workingPlanInfo.workingPlan.planId,
    planPaths,
    simulationTimestamp: rewindInfo.rewindTo,
    fastForwardInfo: currentTimeInfo.fastForwardInfo,
    planMigratedFromVersion,
    planParamsHistoryPreBase: fGet(
      rewindInfo?.planParamsHistoryPreBase ?? null,
    ),
    planParamsHistoryPostBase: workingPlanInfo.planParamsUndoRedoStack.undos,

    updatePlanParams,
    simulationInfoBySrc,
    simulationInfoByMode: {
      mode: 'history',
      actualCurrentTimestamp: currentTimeInfo.currentTimestamp,
    },
    pdfReportInfo,
  }
}

// Info and result contexts are split up because in print view, we have
// the result, but not the info.
const [SimulationInfoContext, useSimulationInfo] =
  createContext<SimulationInfo>('SimulationInfo')

export type SimulationResultInfo = {
  // TODO: Testing. Rename to SimulationResult
  simulationResult: SimulationResult
  simulationIsRunningInfo:
    | {
        isRunning: true
        simulationStartTimestamp: {
          countingFromThisSimulation: number
          countingFromTheFirstDebouncedSimulation: number
        }
      }
    | {
        isRunning: false
      }
}

const [SimulationResultInfoContext, useSimulationResultInfo] =
  createContext<SimulationResultInfo>('SimulationResultInfo')

const [DailyMarketSeriesSrcContext, useDailyMarketSeriesSrc] = createContext<{
  dailyMarketSeriesSrc: DailyMarketSeriesSrc
  setDailyMarketSeriesSrc: (x: DailyMarketSeriesSrc) => void
}>('DailyMarketSeriesSrc')

export const APPLY_DEFAULT_NUM_OF_SIMULATIONS = (x: number | 'default') =>
  x === 'default' ? 1000 : x

export { useDailyMarketSeriesSrc }

export const WithSimulation = React.memo(
  ({ params }: { params: SimulationParams }) => {
    const { nonPlanParams } = useNonPlanParams()
    const numOfSimulationForMonteCarloSampling =
      APPLY_DEFAULT_NUM_OF_SIMULATIONS(
        nonPlanParams.numOfSimulationForMonteCarloSampling,
      )
    const [randomSeed, setRandomSeed] = useState(
      DEFAULT_MONTE_CARLO_SIMULATION_SEED,
    )
    const reRun = useCallback((seed: 'random' | number) => {
      setRandomSeed(
        seed === 'random' ? Math.floor(Math.random() * 1000000000) : seed,
      )
    }, [])
    const { ianaTimezoneName } = useIANATimezoneName()
    const [dailyMarketSeriesSrc, setDailyMarketSeriesSrc] =
      useState<DailyMarketSeriesSrc>({
        type: 'live',
      })

    // Join pre and post base history.
    const planParamsHistoryUpToActualCurrentTimestamp = useMemo(
      () =>
        params.planParamsHistoryPreBase
          ? [
              ...params.planParamsHistoryPreBase,
              ...params.planParamsHistoryPostBase.slice(1), // First item is base which is repeated.
            ]
          : params.planParamsHistoryPostBase,
      [params.planParamsHistoryPreBase, params.planParamsHistoryPostBase],
    )

    const getHistoryDay = useMemo(
      () => _getHistoryDayMemoized(params.planId, ianaTimezoneName),
      [params.planId, ianaTimezoneName],
    )
    const planParamsHistoryDays = useMemo(
      () =>
        new Set(
          planParamsHistoryUpToActualCurrentTimestamp.map((x) =>
            getHistoryDay(x.params.timestamp),
          ),
        ),
      [planParamsHistoryUpToActualCurrentTimestamp, getHistoryDay],
    )

    const { simulationResult, isRunningInfo, planParamsId, planParamsNorm } =
      useSimulator(
        params.planId,
        dailyMarketSeriesSrc,
        planParamsHistoryUpToActualCurrentTimestamp,
        params.simulationTimestamp,
        ianaTimezoneName,
        PERCENTILES,
        numOfSimulationForMonteCarloSampling,
        randomSeed,
      )

    const simulationInfo: SimulationInfo = {
      ...params,
      planParamsNormInstant: planParamsNorm,
      planParamsId,
      planParamsHistoryDays,
      numOfSimulationForMonteCarloSampling,
      reRun,
      randomSeed,
    }

    useShowPDFReportIfNeeded(
      simulationResult,
      _getPlanLabel(simulationInfo.simulationInfoBySrc),
      params.pdfReportInfo,
    )

    const { setGlobalSuspend } = useGlobalSuspenseFallbackContext()
    useEffect(() => {
      setGlobalSuspend(!simulationResult)
    }, [simulationResult, setGlobalSuspend])
    useAssertConst([setGlobalSuspend])

    if (!simulationResult) return <div></div>
    return (
      <DailyMarketSeriesSrcContext.Provider
        value={{ dailyMarketSeriesSrc, setDailyMarketSeriesSrc }}
      >
        <SimulationInfoContext.Provider value={simulationInfo}>
          <SimulationResultInfoContext.Provider
            value={{
              simulationResult,
              simulationIsRunningInfo: isRunningInfo,
            }}
          >
            <Plan />
          </SimulationResultInfoContext.Provider>
        </SimulationInfoContext.Provider>
      </DailyMarketSeriesSrcContext.Provider>
    )
  },
)
export {
  SimulationResultInfoContext,
  useSimulationInfo,
  useSimulationResultInfo,
}

const useShowPDFReportIfNeeded = (
  simulationResult: SimulationResult | null,
  planLabel: string | null,
  pdfReportInfo: SimulationParams['pdfReportInfo'],
) => {
  const isLoggedIn = useUser() !== null
  const { nonPlanParams, setNonPlanParams } = useNonPlanParams()

  const settings = useMemo((): PlanPrintViewSettingsClientSide => {
    const { pdfReportSettings } = nonPlanParams
    return {
      isServerSidePrint: false,
      pageSize: nonPlanParamFns.resolvePDFReportSettingsDefaults.pageSize(
        pdfReportSettings.pageSize,
      ),

      shouldEmbedLink:
        nonPlanParamFns.resolvePDFReportSettingsDefaults.getShouldEmbedLink(
          pdfReportSettings.shouldEmbedLink,
          isLoggedIn,
        ),
      alwaysShowAllMonths: nonPlanParams.dev.alwaysShowAllMonths,
    }
  }, [isLoggedIn, nonPlanParams])

  const handleShowPrintEvent = (simulationResult: SimulationResult) => {
    if (pdfReportInfo.isShowing) return
    const printPlanParams = normalizePlanParamsInverse(
      simulationResult.planParamsNormOfResult,
    )
    const { datingInfo } = simulationResult.planParamsNormOfResult
    const fixed: PlanPrintViewArgs['fixed'] = {
      planLabel,
      datingInfo: datingInfo.isDated
        ? {
            isDatedPlan: true,
            simulationTimestamp: datingInfo.nowAsTimestamp,
            ianaTimezoneName: fGet(
              simulationResult.ianaTimezoneNameIfDatedPlanOfResult,
            ),
          }
        : {
            isDatedPlan: false,
            timestampForMarketData: getNYZonedTime
              .fromObject(datingInfo.marketDataAsOfEndOfDayInNY)
              .endOf('day')
              .toMillis(),
          },
      dailyMarketSeriesSrc: simulationResult.dailyMarketSeriesSrcOfResult,
      percentiles: simulationResult.percentilesOfResult,
      currentPortfolioBalanceAmount:
        simulationResult.portfolioBalanceEstimationByDated.currentBalance,
      planParams: printPlanParams,
      numOfSimulationForMonteCarloSampling:
        simulationResult.numOfSimulationForMonteCarloSamplingOfResult,
      randomSeed: simulationResult.randomSeedOfResult,
    }
    pdfReportInfo.show({
      fixed,
      settings,
      simulationResult,
      updateSettings: (settings) => {
        const clone = cloneJSON(nonPlanParams)
        clone.pdfReportSettings = {
          pageSize: settings.pageSize,
          shouldEmbedLink: settings.shouldEmbedLink ? 'yes' : 'no',
        }
        setNonPlanParams(clone)
      },
    })
  }
  const handleShowPrintEventRef = useRef(handleShowPrintEvent)
  useInsertionEffect(() => {
    handleShowPrintEventRef.current = handleShowPrintEvent
  }, [handleShowPrintEvent])
  const shouldShowPrint = useURLParam('pdf-report') === 'true'
  useLayoutEffect(() => {
    if (shouldShowPrint && simulationResult) {
      handleShowPrintEventRef.current(simulationResult)
    }
  }, [shouldShowPrint, simulationResult])

  const handleUpdateSettingsEvent = (
    settings: PlanPrintViewSettingsClientSide,
  ) => {
    if (pdfReportInfo.isShowing) pdfReportInfo.onSettings(settings)
  }
  const handleUpdateSettingsEventRef = useRef(handleUpdateSettingsEvent)
  handleUpdateSettingsEventRef.current = handleUpdateSettingsEvent
  useLayoutEffect(() => {
    handleUpdateSettingsEventRef.current(settings)
  }, [settings])
}

const _getHistoryDayMemoized = (planId: string, ianaTimezoneName: string) => {
  const cache = new Map<number, number>()
  const getZonedTime = getZonedTimeFns(ianaTimezoneName)
  return (timestamp: number) => {
    if (!cache.has(timestamp)) {
      const value = getZonedTime(timestamp).startOf('day').toMillis()
      cache.set(timestamp, value)
    }
    return cache.get(timestamp)!
  }
}

const _getPlanLabel = (
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
) => {
  switch (simulationInfoBySrc.src) {
    case 'link':
    case 'localMain':
      return null
    case 'file':
      return PlanFileDataFns.labelFromFilename(
        simulationInfoBySrc.plan.filename,
      )
    case 'server':
      return simulationInfoBySrc.plan.label ?? null
    default:
      noCase(simulationInfoBySrc)
  }
}
