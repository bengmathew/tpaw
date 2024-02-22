import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  MarketData,
  PlanParams,
  PlanParamsChangeActionCurrent,
  PlanParamsHistoryFns,
  SomePlanParamsVersion,
  assert,
  block,
  fGet,
  getDefaultPlanParams,
  noCase,
  nonPlanParamFns,
} from '@tpaw/common'
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  PlanParamsExtended,
  extendPlanParams,
} from '../../../UseSimulator/ExtentPlanParams'
import {
  PlanParamsProcessed,
  processPlanParams,
} from '../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { useSimulator } from '../../../UseSimulator/UseSimulator'
import { createContext } from '../../../Utils/CreateContext'

import cloneJSON from 'fast-json-clone'
import _ from 'lodash'
import React from 'react'
import { appPaths } from '../../../AppPaths'
import { SimulationResult } from '../../../UseSimulator/Simulator/Simulator'
import { infoToast } from '../../../Utils/CustomToasts'
import { UnionToIntersection } from '../../../Utils/UnionToIntersection'
import { useURLParam } from '../../../Utils/UseURLParam'
import { User, useUser } from '../../App/WithUser'
import { getMarketDataForTime } from '../../Common/GetMarketData'
import { Plan } from '../Plan/Plan'
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
import * as Rust from '@tpaw/simulator'
import { PlanParamsNormalized } from '../../../UseSimulator/NormalizePlanParams'

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

  currentTimestamp: number
  fastForwardInfo: CurrentTimeInfo['fastForwardInfo']

  defaultPlanParams: PlanParams
  currentMarketData: MarketData.Data[0] & {
    timestampMSForHistoricalReturns: number
  }

  currentPortfolioBalanceInfo: ReturnType<
    typeof CurrentPortfolioBalance.cutInfo
  >

  planParamsId: string
  planParams: PlanParams
  planParamsExt: PlanParamsExtended
  planParamsNorm: PlanParamsNormalized
  planParamsProcessed: PlanParamsProcessed
  planMigratedFromVersion: SomePlanParamsVersion

  updatePlanParams: UpdatePlanParams

  simulationInfoBySrc:
    | {
        src: 'localMain'
        reset: () => void
      }
    | {
        src: 'server'
        plan: User['plans'][0]
        historyStatus: 'fetching' | 'fetched' | 'failed'
        syncState: ServerSyncState
      }
    | {
        src: 'link'
        isModified: boolean
        reset: () => void
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
        planParamsHistoryEndOfDays: Set<number>
        setRewindTo: (timestamp: number) => 'withinRange' | 'fixedToRange'
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
  | 'defaultPlanParams'
  | 'currentMarketData'
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
    setRewindTo: (timestamp: number) => 'withinRange' | 'fixedToRange'
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

  const planParamsHistoryEndOfDays = useMemo(
    () =>
      new Set(
        planParamsHistory.map((x) =>
          getZonedTime(x.params.timestamp).endOf('day').toMillis(),
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
      planParamsHistoryEndOfDays,
      setRewindTo: rewindInfo.setRewindTo,
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

    const defaultPlanParams = useMemo(
      () => getDefaultPlanParams(currentTimestamp, ianaTimezoneName),
      [currentTimestamp, ianaTimezoneName],
    )

    const currentMarketData = useMemo(
      () => ({
        ...getMarketDataForTime(currentTimestamp, marketData),
        timestampMSForHistoricalReturns: currentTimestamp,
      }),
      [currentTimestamp, marketData],
    )

    const { currentPortfolioBalanceInfo, planParamsProcessed, planParamsExt } =
      useMemo(() => {
        const currentPortfolioBalanceInfo = CurrentPortfolioBalance.cutInfo(
          currentTimestamp,
          {
            preBase: currentPortfolioBalanceInfoPreBase,
            postBase: currentPortfolioBalanceInfoPostBase,
          },
        )
        const planParamsExt = extendPlanParams(
          planParams,
          currentTimestamp,
          ianaTimezoneName,
        )

        const planParamsProcessed = processPlanParams(
          planParamsExt,
          CurrentPortfolioBalance.get(currentPortfolioBalanceInfo),
          currentMarketData,
        )

        return {
          currentPortfolioBalanceInfo,
          planParamsExt,
          planParamsProcessed,
        }
      }, [
        currentMarketData,
        currentPortfolioBalanceInfoPostBase,
        currentPortfolioBalanceInfoPreBase,
        currentTimestamp,
        ianaTimezoneName,
        planParams,
      ])

    const {
      result: simulationResult,
      resultIsCurrent: simulationResultIsCurrent,
    } = useSimulator(
      planParamsProcessed,
      planParamsExt,
      numOfSimulationForMonteCarloSampling,
      randomSeed,
    )

    const simulationInfo: SimulationInfo | null = simulationResult
      ? {
          ...params,
          defaultPlanParams,
          currentMarketData,
          planParamsExt,
          planParamsNorm: planParamsProcessed.planParamsNorm,
          planParamsProcessed,
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
  const { ianaTimezoneName } = useIANATimezoneName()
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

      embeddedLinkType:
        nonPlanParamFns.resolvePDFReportSettingsDefaults.embeddedLinkType(
          pdfReportSettings.embeddedLinkType,
          isLoggedIn,
        ),
      alwaysShowAllMonths: nonPlanParams.dev.alwaysShowAllMonths,
    }
  }, [isLoggedIn, nonPlanParams])

  const handleShowPrintEvent = () => {
    assert(simulationInfo && !pdfReportInfo.isShowing)

    const printPlanParams = cloneJSON(simulationInfo.planParams)
    printPlanParams.timestamp = simulationInfo.currentTimestamp
    printPlanParams.wealth.portfolioBalance = {
      updatedHere: true,
      amount: CurrentPortfolioBalance.get(
        simulationInfo.currentPortfolioBalanceInfo,
      ),
    }
    const fixed: PlanPrintViewArgs['fixed'] = {
      planLabel: block(() => {
        switch (simulationInfo.simulationInfoBySrc.src) {
          case 'link':
          case 'localMain':
            return null
          case 'server':
            return simulationInfo.simulationInfoBySrc.plan.label ?? null
          default:
            noCase(simulationInfo.simulationInfoBySrc)
        }
      }),
      planParams: printPlanParams,
      marketData: simulationInfo.currentMarketData,
      numOfSimulationForMonteCarloSampling:
        simulationInfo.numOfSimulationForMonteCarloSampling,
      ianaTimezoneName,
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
          embeddedLinkType: settings.embeddedLinkType,
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
