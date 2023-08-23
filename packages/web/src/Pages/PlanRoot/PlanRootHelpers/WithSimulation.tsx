import {
  MarketData,
  PlanParams,
  PlanParamsChangeAction,
  PlanParamsHistoryFns,
  SomePlanParamsVersion,
  assert,
  block,
  fGet,
  getDefaultPlanParams,
} from '@tpaw/common'
import { useCallback, useMemo } from 'react'
import {
  PlanParamsExtended,
  extendPlanParams,
} from '../../../TPAWSimulator/ExtentPlanParams'
import {
  PlanParamsProcessed,
  processPlanParams,
} from '../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import {
  UseTPAWWorkerResult,
  useTPAWWorker,
} from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import { createContext } from '../../../Utils/CreateContext'

import _ from 'lodash'
import React from 'react'
import { appPaths } from '../../../AppPaths'
import { infoToast } from '../../../Utils/CustomToasts'
import { UnionToIntersection } from '../../../Utils/UnionToIntersection'
import { User } from '../../App/WithUser'
import { getMarketDataForTime } from '../../Common/GetMarketData'
import { Plan } from '../Plan/Plan'
import { CurrentPortfolioBalance } from './CurrentPortfolioBalance'
import { CurrentTimeInfo } from './UseCurrentTime'
import { WorkingPlanInfo } from './UseWorkingPlan'
import { useMarketData } from './WithMarketData'
import { useIANATimezoneName } from './WithNonPlanParams'

type UpdatePlanParamsFromAction<T> = UnionToIntersection<
  T extends PlanParamsChangeAction
    ? (x: T['type'], v: T['value']) => void
    : never
>
export type UpdatePlanParams =
  UpdatePlanParamsFromAction<PlanParamsChangeAction>

export type SimulationInfo = {
  planId: string
  planPaths: (typeof appPaths)['plan']

  currentTimestamp: number
  fastForwardInfo: CurrentTimeInfo['fastForwardInfo']

  defaultPlanParams: PlanParams
  currentMarketData: MarketData.Data[0]

  currentPortfolioBalanceInfo: ReturnType<typeof CurrentPortfolioBalance.cutInfo>

  planParamsId: string
  planParams: PlanParams
  planParamsExt: PlanParamsExtended
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
        isSyncing: boolean
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

  tpawResult: UseTPAWWorkerResult
  reRun: () => void
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
  | 'planParamsProcessed'
  | 'tpawResult'
  | 'reRun'
  | 'currentPortfolioBalanceInfo'
> & {
  currentPortfolioBalanceInfoPreBase: CurrentPortfolioBalance.ByMonthInfo | null
  currentPortfolioBalanceInfoPostBase: CurrentPortfolioBalance.Info
}

export const useSimulationParamsForPlanMode = (
  planPaths: SimulationInfo['planPaths'],
  currentTimeInfo: CurrentTimeInfo,
  workingPlanInfo: WorkingPlanInfo,
  planMigratedFromVersion: SomePlanParamsVersion,
  currentPortfolioBalanceInfoPreBase: CurrentPortfolioBalance.ByMonthInfo | null,
  simulationInfoBySrc: SimulationInfo['simulationInfoBySrc'],
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
  }
}

const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')
export const WithSimulation = React.memo(
  ({ params }: { params: SimulationParams }) => {
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
      () => getMarketDataForTime(currentTimestamp, marketData),
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

    const { result: tpawResult, reRun } = useTPAWWorker(
      planParamsProcessed,
      planParamsExt,
    )

    return (
      tpawResult && (
        <Context.Provider
          value={{
            ...params,
            defaultPlanParams,
            currentMarketData,
            planParamsExt,
            planParamsProcessed,
            currentPortfolioBalanceInfo,
            tpawResult,
            reRun,
          }}
        >
          <Plan />
        </Context.Provider>
      )
    )
  },
)
export { useSimulation }
