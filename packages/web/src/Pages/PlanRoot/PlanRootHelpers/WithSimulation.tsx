import {
  CalendarDay,
  CalendarMonthFns,
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  PlanParams,
  PlanParamsChangeActionCurrent,
  PlanParamsHistoryFns,
  SomePlanParamsVersion,
  assert,
  block,
  fGet,
  getNYZonedTime,
  noCase,
  nonPlanParamFns,
} from '@tpaw/common'
import cloneJSON from 'fast-json-clone'
import _ from 'lodash'
import React, {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { appPaths } from '../../../AppPaths'
import {
  PlanParamsNormalized,
  normalizePlanParams,
} from '../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { normalizePlanParamsInverse } from '../../../UseSimulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { CallRust } from '../../../UseSimulator/PlanParamsProcessed/CallRust'
import { processPlanParams } from '../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { SimulationResult } from '../../../UseSimulator/Simulator/Simulator'
import { useSimulator } from '../../../UseSimulator/UseSimulator'
import { createContext } from '../../../Utils/CreateContext'
import { infoToast } from '../../../Utils/CustomToasts'
import { UnionToIntersection } from '../../../Utils/UnionToIntersection'
import { useURLParam } from '../../../Utils/UseURLParam'
import { User, useUser } from '../../App/WithUser'
import { getMarketDataForTime } from '../../Common/GetMarketData'
import { Plan } from '../Plan/Plan'
import { PlanFileData, PlanFileDataFns } from '../PlanRootFile/PlanFileData'
import { ServerSyncState } from '../PlanServerImpl/UseServerSyncPlan'
import { CurrentPortfolioBalance } from './CurrentPortfolioBalance'
import {
  PlanPrintViewArgs,
  PlanPrintViewSettingsClientSide,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintView/PlanPrintViewArgs'
import { CurrentTimeInfo } from './UseCurrentTime'
import { WorkingPlanInfo } from './UseWorkingPlan'
import { useMarketData } from './WithMarketData'
import { useIANATimezoneName, useNonPlanParams } from './WithNonPlanParams'
import { CalendarDayFns } from '../../../Utils/CalendarDayFns'

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

  // defaultPlanParams: PlanParams

  currentPortfolioBalanceInfo:
    | {
        isDatedPlan: true
        info: CurrentPortfolioBalance.CutInfo
      }
    | { isDatedPlan: false; amount: number }

  planParamsId: string
  planParamsNorm: PlanParamsNormalized
  planMigratedFromVersion: SomePlanParamsVersion

  updatePlanParams: UpdatePlanParams

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
        planParamsHistory: { id: string; params: PlanParams }[]
        planParamsHistoryDays: Set<number>
      }

  simulationResult: SimulationResult
  simulationResultIsCurrent: boolean
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
  | 'nowAsCalendarDay'
  | 'defaultPlanParams'
  | 'planParamsExt'
  | 'planParamsNorm'
  | 'planParamsProcessed'
  | 'simulationResult'
  | 'simulationResultIsCurrent'
  | 'reRun'
  | 'currentPortfolioBalanceInfo'
  | 'numOfSimulationForMonteCarloSampling'
  | 'randomSeed'
> & {
  currentTimestamp: number
  planParams: PlanParams
  currentPortfolioBalanceInfoPreBase: CurrentPortfolioBalance.ByMonthInfo | null
  currentPortfolioBalanceInfoPostBase: CurrentPortfolioBalance.Info
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

export const useSimulationParamsForPlanMode = (
  planPaths: SimulationInfo['planPaths'],
  currentTimeInfo: CurrentTimeInfo,
  workingPlanInfo: WorkingPlanInfo,
  // TODO: This is not necessary since PlanParamsHistory contains unmigrated version.
  planMigratedFromVersion: SomePlanParamsVersion,
  currentPortfolioBalanceInfoPreBase: CurrentPortfolioBalance.ByMonthInfo | null,
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
  pdfReportInfo: SimulationParams['pdfReportInfo'],
): SimulationParams => ({
  planId: workingPlanInfo.workingPlan.planId,
  planPaths,

  currentTimestamp: currentTimeInfo.currentTimestamp,
  fastForwardInfo: currentTimeInfo.fastForwardInfo,

  ...block(() => {
    const { id, params } = fGet(
      _.last(workingPlanInfo.planParamsUndoRedoStack.undos),
    )
    return { planParamsId: id, planParams: params }
  }),
  planMigratedFromVersion,

  currentPortfolioBalanceInfoPreBase,
  currentPortfolioBalanceInfoPostBase:
    workingPlanInfo.currentPortfolioBalanceInfoPostBase,

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
    currentPortfolioBalanceInfoPreBase: CurrentPortfolioBalance.ByMonthInfo
    planParamsHistoryPreBase: { id: string; params: PlanParams }[]
  } | null,
  planPaths: SimulationInfo['planPaths'],
  currentTimeInfo: CurrentTimeInfo,
  workingPlanInfo: WorkingPlanInfo,
  planMigratedFromVersion: SomePlanParamsVersion,
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
  pdfReportInfo: SimulationParams['pdfReportInfo'],
): SimulationParams | null => {
  // Note. Avoid computation if returning null. We don't want to slow things
  // down when not in history mode.
  const { ianaTimezoneName, getZonedTime } = useIANATimezoneName()

  const updatePlanParams = useCallback(
    () => infoToast('Cannot update plan in history mode.'),
    [],
  )
  const returnNull = rewindInfo === null
  const rewindTo = rewindInfo?.rewindTo ?? null

  const currentPortfolioBalanceInfoPreBase =
    rewindInfo?.currentPortfolioBalanceInfoPreBase ?? null
  const planParamsHistoryPreBase = rewindInfo?.planParamsHistoryPreBase ?? null

  const planParamsHistory = useMemo(() => {
    if (returnNull) return []
    assert(planParamsHistoryPreBase)
    assert(
      fGet(_.last(planParamsHistoryPreBase)).id ===
        fGet(_.first(workingPlanInfo.planParamsUndoRedoStack.undos)).id,
    )
    const unfiltered = [
      ...planParamsHistoryPreBase,
      ...workingPlanInfo.planParamsUndoRedoStack.undos.slice(1),
    ]
    const { idsToDelete } = PlanParamsHistoryFns.filterForLastChangePerDay({
      ianaTimezoneName,
      planParamsHistory: unfiltered.map((x) => ({
        planParamsChangeId: x.id,
        timestamp: new Date(x.params.timestamp),
      })),
      intersectWithIds: null,
    })
    return unfiltered.filter((x) => !idsToDelete.has(x.id))
  }, [
    ianaTimezoneName,
    planParamsHistoryPreBase,
    returnNull,
    workingPlanInfo.planParamsUndoRedoStack.undos,
  ])

  const planParamsHistoryDays = useMemo(
    () =>
      new Set(
        planParamsHistory.map((x) =>
          getZonedTime(x.params.timestamp).startOf('day').toMillis(),
        ),
      ),
    [planParamsHistory, getZonedTime],
  )

  const headIndex = useMemo(() => {
    if (returnNull) return 0
    assert(rewindTo !== null)
    return (
      _.sortedLastIndexBy<{ params: { timestamp: number } }>(
        planParamsHistory,
        { params: { timestamp: rewindTo } },
        (x) => x.params.timestamp,
      ) - 1
    )
  }, [planParamsHistory, returnNull, rewindTo])

  if (returnNull) return null

  assert(rewindTo !== null)

  return {
    planId: workingPlanInfo.workingPlan.planId,
    planPaths,

    currentTimestamp: rewindTo,
    fastForwardInfo: currentTimeInfo.fastForwardInfo,

    planParamsId: planParamsHistory[headIndex].id,
    planParams: planParamsHistory[headIndex].params,
    planMigratedFromVersion,

    currentPortfolioBalanceInfoPreBase,
    currentPortfolioBalanceInfoPostBase:
      workingPlanInfo.currentPortfolioBalanceInfoPostBase,

    updatePlanParams,
    simulationInfoBySrc,
    simulationInfoByMode: {
      mode: 'history',
      actualCurrentTimestamp: currentTimeInfo.currentTimestamp,
      planParamsHistory,
      planParamsHistoryDays,
    },
    pdfReportInfo,
  }
}

const [SimulationInfoContext, useSimulation] =
  createContext<SimulationInfo>('SimulationInfo')
const [SimulationResultContext, useSimulationResult] =
  createContext<SimulationResult>('SimulationResult')

// Get a random number.

const _newRandomSeed = () => {
  return Math.floor(Math.random() * 1000000000)
}
let globalRandomSeed = DEFAULT_MONTE_CARLO_SIMULATION_SEED

export const WithSimulation = React.memo(
  ({ params }: { params: SimulationParams }) => {
    const { nonPlanParams } = useNonPlanParams()
    const { numOfSimulationForMonteCarloSampling } = nonPlanParams
    const [randomSeed, setRandomSeed] = useState(globalRandomSeed)
    const reRun = useCallback((seed: 'random' | number) => {
      globalRandomSeed = seed === 'random' ? _newRandomSeed() : seed
      setRandomSeed(globalRandomSeed)
    }, [])

    const {
      planParams,
      currentTimestamp,
      currentPortfolioBalanceInfoPreBase,
      currentPortfolioBalanceInfoPostBase,
    } = params
    const { ianaTimezoneName } = useIANATimezoneName()
    const { marketData } = useMarketData()

    const {
      currentPortfolioBalanceInfo,
      planParamsNorm,
      currentMarketData,
      planParamsProcessed,
      planParamsRust,
    } = useMemo(() => {
      const planParamsNorm = normalizePlanParams(planParams, {
        timestamp: currentTimestamp,
        calendarDay: CalendarDayFns.fromTimestamp(
          currentTimestamp,
          ianaTimezoneName,
        ),
      })

      const currentMarketData = {
        ...getMarketDataForTime(
          planParamsNorm.datingInfo.timestampForMarketData,
          marketData,
        ),
        timestampForMarketData:
          planParamsNorm.datingInfo.timestampForMarketData,
      }

      const currentPortfolioBalanceInfo = planParamsNorm.wealth.portfolioBalance
        .isDatedPlan
        ? ({
            isDatedPlan: true,
            info: CurrentPortfolioBalance.cutInfo(currentTimestamp, {
              preBase: currentPortfolioBalanceInfoPreBase,
              postBase: currentPortfolioBalanceInfoPostBase,
            }),
          } as const)
        : ({
            isDatedPlan: false,
            amount: planParamsNorm.wealth.portfolioBalance.amount,
          } as const)

      const planParamsRust = CallRust.getPlanParamsRust(planParamsNorm)
      const planParamsProcessed = processPlanParams(
        planParamsNorm,
        currentMarketData,
      )

      return {
        currentPortfolioBalanceInfo,
        currentMarketData,
        planParamsRust,
        planParamsNorm,
        planParamsProcessed,
      }
    }, [
      currentTimestamp,
      ianaTimezoneName,
      planParams,
      marketData,
      currentPortfolioBalanceInfoPreBase,
      currentPortfolioBalanceInfoPostBase,
    ])

    const {
      result: simulationResult,
      resultIsCurrent: simulationResultIsCurrent,
    } = useSimulator(
      currentPortfolioBalanceInfo.isDatedPlan
        ? CurrentPortfolioBalance.getAmountInfo(
            currentPortfolioBalanceInfo.info,
          ).amount
        : currentPortfolioBalanceInfo.amount,
      planParamsRust,
      currentMarketData,
      planParamsNorm,
      planParamsProcessed,
      numOfSimulationForMonteCarloSampling,
      randomSeed,
    )

    const simulationInfo: SimulationInfo | null = simulationResult
      ? {
          ...params,
          planParamsNorm,
          currentPortfolioBalanceInfo,
          simulationResult,
          simulationResultIsCurrent,
          numOfSimulationForMonteCarloSampling,
          reRun,
          randomSeed,
        }
      : null

    useShowPDFReportIfNeeded(simulationInfo, params.pdfReportInfo)
    return (
      simulationInfo && (
        <SimulationInfoContext.Provider value={simulationInfo}>
          <SimulationResultContext.Provider
            value={simulationInfo.simulationResult}
          >
            <Plan />
          </SimulationResultContext.Provider>
        </SimulationInfoContext.Provider>
      )
    )
  },
)
export { SimulationResultContext, useSimulation, useSimulationResult }

const useShowPDFReportIfNeeded = (
  simulationInfo: SimulationInfo | null,
  pdfReportInfo: SimulationParams['pdfReportInfo'],
) => {
  const isLoggedIn = useUser() !== null
  const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
  const shouldShowPrint =
    useURLParam('pdf-report') === 'true' &&
    !pdfReportInfo.isShowing &&
    !!simulationInfo &&
    simulationInfo.simulationResultIsCurrent

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

  const handleShowPrintEvent = () => {
    assert(simulationInfo && !pdfReportInfo.isShowing)

    const printPlanParams = normalizePlanParamsInverse(
      simulationInfo.planParamsNorm,
    )
    const { datingInfo } = simulationInfo.planParamsNorm
    const fixed: PlanPrintViewArgs['fixed'] = {
      planLabel: block(() => {
        switch (simulationInfo.simulationInfoBySrc.src) {
          case 'link':
          case 'localMain':
            return null
          case 'file':
            return PlanFileDataFns.labelFromFilename(
              simulationInfo.simulationInfoBySrc.plan.filename,
            )
          case 'server':
            return simulationInfo.simulationInfoBySrc.plan.label ?? null
          default:
            noCase(simulationInfo.simulationInfoBySrc)
        }
      }),
      datingInfo: datingInfo.isDated
        ? {
            isDatedPlan: true,
            nowAsTimestamp: datingInfo.nowAsTimestamp,
            nowAsCalendarDay: datingInfo.nowAsCalendarDay,
          }
        : {
            isDatedPlan: false,
            timestampForMarketData: getNYZonedTime
              .fromObject(datingInfo.marketDataAsOfEndOfDayInNY)
              .endOf('day')
              .toMillis(),
          },
      currentPortfolioBalanceAmount: simulationInfo.currentPortfolioBalanceInfo
        .isDatedPlan
        ? CurrentPortfolioBalance.getAmountInfo(
            simulationInfo.currentPortfolioBalanceInfo.info,
          ).amount
        : simulationInfo.currentPortfolioBalanceInfo.amount,
      planParams: printPlanParams,
      numOfSimulationForMonteCarloSampling:
        simulationInfo.numOfSimulationForMonteCarloSampling,
      randomSeed: simulationInfo.randomSeed,
    }

    pdfReportInfo.show({
      fixed,
      settings,
      simulationResult: simulationInfo.simulationResult,
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
  handleShowPrintEventRef.current = handleShowPrintEvent
  useLayoutEffect(() => {
    if (shouldShowPrint) handleShowPrintEventRef.current()
  }, [shouldShowPrint])

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
