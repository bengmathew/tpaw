import * as Sentry from '@sentry/nextjs'
import {
  CalendarDay,
  FGet,
  PlanParams,
  PlanParamsChangeAction,
  SomePlanParams,
  SomePlanParamsVersion,
  assert,
  assertFalse,
  block,
  fGet,
  letIn,
  noCase,
  planParamsMigrate,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { appPaths } from '../../../AppPaths'
import { sendAnalyticsEvent } from '../../../Utils/SendAnalyticsEvent'
import { useNavGuard } from '../../../Utils/UseNavGuard'
import { useURLParam } from '../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { Spinner } from '../../../Utils/View/Spinner'
import { useUser } from '../../App/WithUser'
import { CenteredModal } from '../../Common/Modal/CenteredModal'
import { useCurrentTime } from '../PlanRootHelpers/UseCurrentTime'
import { useWorkingPlan } from '../PlanRootHelpers/UseWorkingPlan'
import { useIANATimezoneName } from '../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForServerSrc,
  SimulationParams,
  WithSimulation,
  useSimulationParamsForHistoryMode,
  useSimulationParamsForPlanMode,
} from '../PlanRootHelpers/WithSimulation'
import { PlanRootServerQuery$data } from '../PlanRootServer/__generated__/PlanRootServerQuery.graphql'
import { PlanServerImplSyncState } from './PlanServerImplSyncState'
import { useServerHistoryPreBase } from './UseServerHistoryFromStart'
import { useServerSyncPlan } from './UseServerSyncPlan'

type _Props = {
  plan: FGet<Exclude<PlanRootServerQuery$data['user'], undefined>['plan']>
  planPaths: (typeof appPaths)['plan']
  pdfReportInfo: SimulationParams['pdfReportInfo']
  reload: () => void
}

export const PlanServerImpl = React.memo((props: _Props) => {
  const { plan } = props
  const [time, setTime] = useState(() => Date.now())
  const firstParamsTime = useMemo(
    () =>
      planParamsMigrate(
        JSON.parse(
          fGet(_.last(plan.planParamsPostBase)).params,
        ) as SomePlanParams,
      ).timestamp,
    [plan],
  )
  assert(time >= firstParamsTime - 3 * 1000)

  // Wait for clock to catch up to server.
  const shouldWait = time < firstParamsTime
  useEffect(() => {
    if (!shouldWait) return
    const timeout = setTimeout(() => setTime(Date.now()), 100)
    return () => clearTimeout(timeout)
  }, [shouldWait])

  const createEvent = () => {
    sendAnalyticsEvent('planServerImplCreation', {
      clockSkew: time - firstParamsTime,
    })
  }
  const createEventRef = useRef(createEvent)
  createEventRef.current = createEvent
  useEffect(() => {
    createEventRef.current()
  }, [])

  if (shouldWait) return <></>
  return <_Body {...props} />
})

const _Body = React.memo(
  ({
    plan: serverPlanIn,
    planPaths,
    pdfReportInfo,
    reload,
  }: {
    plan: FGet<Exclude<PlanRootServerQuery$data['user'], undefined>['plan']>
    planPaths: (typeof appPaths)['plan']
    pdfReportInfo: SimulationParams['pdfReportInfo']
    reload: () => void
  }) => {
    const user = fGet(useUser())
    const { getZonedTime } = useIANATimezoneName()
    const [startingServerPlan] = useState(() => ({
      planId: serverPlanIn.id,
      lastSyncAt: serverPlanIn.lastSyncAt,
      planParamsPostBase: serverPlanIn.planParamsPostBase.map((x) => {
        const paramsUnmigrated: SomePlanParams = JSON.parse(x.params)
        return {
          id: x.id,
          change: JSON.parse(x.change) as PlanParamsChangeAction,
          paramsUnmigrated,
          params: planParamsMigrate(paramsUnmigrated),
        }
      }),
      reverseHeadIndex: serverPlanIn.reverseHeadIndex,
    }))
    const [startTime] = useState(() => Date.now())
    const firstParamsTime = fGet(_.last(startingServerPlan.planParamsPostBase))
      .params.timestamp
    if (startTime < firstParamsTime) {
      throw new Error(
        `startTime-firstParamsTime=${
          startTime - firstParamsTime
        }  is less than 0.`,
      )
    }

    const [planMigratedFromVersion] = useState<SomePlanParamsVersion>(() =>
      letIn(
        JSON.parse(
          fGet(_.last(serverPlanIn.planParamsPostBase)).params,
        ) as SomePlanParams,
        (lastPlanParams) => ('v' in lastPlanParams ? lastPlanParams.v : 1),
      ),
    )

    const [serverPlan, setServerPlan] = useState(
      () =>
        ({
          planId: startingServerPlan.planId,
          lastSyncAt: startingServerPlan.lastSyncAt,
          planParamsPostBaseIds: startingServerPlan.planParamsPostBase.map(
            (x) => x.id,
          ) as ReadonlyArray<string>,
          reverseHeadIndex: startingServerPlan.reverseHeadIndex,
        }) as const,
    )
    const planId = serverPlan.planId

    const currentTimeInfo = useCurrentTime({ planId, startTime })

    assert(
      currentTimeInfo.currentTimestamp >=
        fGet(_.last(startingServerPlan.planParamsPostBase)).params.timestamp,
    )

    const workingPlanInfo = useWorkingPlan(
      currentTimeInfo,
      startingServerPlan,
      planPaths,
    )
    const serverSyncState = useServerSyncPlan(
      user.id,
      serverPlan,
      workingPlanInfo.workingPlan,
      setServerPlan,
    )
    const isSyncing = serverSyncState.type !== 'synced'

    const serverHistoryPreBaseInfo = useServerHistoryPreBase(
      planId,
      fGet(_.first(workingPlanInfo.planParamsUndoRedoStack.undos)),
    )

    const handleRebase = (
      currPreBase: { id: string; params: PlanParams }[],
      workingPlanRebase: Exclude<(typeof workingPlanInfo)['rebase'], null>,
    ) => {
      const cutAndBase = workingPlanRebase({ hard: false })
      assert(cutAndBase.length > 0)
      setServerPlan((serverPlan) => {
        assert(
          cutAndBase.every(
            (x, i) => x.id === serverPlan.planParamsPostBaseIds[i],
          ),
        )
        return {
          ...serverPlan,
          planParamsPostBaseIds: serverPlan.planParamsPostBaseIds.slice(
            cutAndBase.length - 1,
          ),
        }
      })
      serverHistoryPreBaseInfo.rebase(currPreBase, cutAndBase)
    }
    const handleRebaseRef = useRef(handleRebase)
    handleRebaseRef.current = handleRebase
    useEffect(() => {
      if (!workingPlanInfo.rebase) return
      if (serverHistoryPreBaseInfo.state.type !== 'fetched') return
      if (isSyncing) return
      handleRebaseRef.current(
        serverHistoryPreBaseInfo.state.planParamsHistory,
        workingPlanInfo.rebase,
      )
    }, [
      serverHistoryPreBaseInfo.state.planParamsHistory,
      serverHistoryPreBaseInfo.state.type,
      workingPlanInfo.rebase,
      isSyncing,
    ])

    if (serverHistoryPreBaseInfo.state.type === 'fetched') {
      assert(
        fGet(_.last(serverHistoryPreBaseInfo.state.planParamsHistory)).id ===
          fGet(_.first(workingPlanInfo.workingPlan.planParamsPostBase)).id,
      )
    }

    const urlUpdater = useURLUpdater()
    const rewindToStr = useURLParam('rewindTo')

    const setRewindTo = useCallback(
      (rewindTo: 'lastUpdate' | CalendarDay | null) => {
        const url = new URL(window.location.href)
        if (rewindTo === null) {
          url.searchParams.delete('rewindTo')
        } else {
          url.searchParams.set(
            'rewindTo',
            rewindTo === 'lastUpdate'
              ? rewindTo
              : `${rewindTo.day}-${rewindTo.month}-${rewindTo.year}`,
          )
        }
        urlUpdater.push(url)
      },
      [urlUpdater],
    )

    const rewindInfo = block(() => {
      const rewindToTimestamp = block(() => {
        if (!rewindToStr) return null
        if (rewindToStr === 'lastUpdate') {
          return fGet(_.last(workingPlanInfo.planParamsUndoRedoStack.undos))
            .params.timestamp
        }
        const parts = rewindToStr.split('-')
        if (parts.length !== 3) return null
        const [day, month, year] = parts.map((x) => parseInt(x))
        if (isNaN(year) || isNaN(month) || isNaN(day)) return null
        return Math.min(
          getZonedTime.fromObject({ year, month, day }).endOf('day').toMillis(),
          currentTimeInfo.currentTimestamp,
        )
      })

      if (rewindToTimestamp === null) return null
      switch (serverHistoryPreBaseInfo.state.type) {
        case 'failed':
          assertFalse()
        case 'fetching':
          return { type: 'fetching' as const }
        case 'fetched':
          const preBase = serverHistoryPreBaseInfo.state
          if (
            rewindToTimestamp < preBase.planParamsHistory[0].params.timestamp
          ) {
            return null
          }

          return {
            type: 'fetched' as const,
            rewindTo: rewindToTimestamp,
            currentPortfolioBalanceInfoPreBase:
              preBase.currentPortfolioBalanceByMonthInfo,
            planParamsHistoryPreBase: preBase.planParamsHistory,
          }
      }
    })

    const simulationInfoBySrc: SimulationInfoForServerSrc = {
      src: 'server',
      plan: serverPlanIn,
      historyStatus: serverHistoryPreBaseInfo.state.type,
      syncState: serverSyncState,
      setRewindTo,
      reload,
    }

    const simulationParamsForPlanMode = useSimulationParamsForPlanMode(
      planPaths,
      currentTimeInfo,
      workingPlanInfo,
      planMigratedFromVersion,
      serverHistoryPreBaseInfo.state.type === 'fetched'
        ? serverHistoryPreBaseInfo.state.currentPortfolioBalanceByMonthInfo
        : serverHistoryPreBaseInfo.state.type === 'fetching' ||
            serverHistoryPreBaseInfo.state.type === 'failed'
          ? null
          : noCase(serverHistoryPreBaseInfo.state),
      simulationInfoBySrc,
      pdfReportInfo,
    )

    const simulationParamsForHistoryMode = useSimulationParamsForHistoryMode(
      rewindInfo && rewindInfo.type === 'fetched' ? rewindInfo : null,
      planPaths,
      currentTimeInfo,
      workingPlanInfo,
      planMigratedFromVersion,
      simulationInfoBySrc,
      pdfReportInfo,
    )

    const { navGuardState, resetNavGuardState } = useNavGuard(
      isSyncing,
      planPaths,
    )
    const [showSyncState, setShowSyncState] = useState(false)
    useEffect(() => {
      if (showSyncState) return
      if (
        serverSyncState.type === 'waitDueToError' &&
        (serverSyncState.waitEndTime === 'never' ||
          serverSyncState.failures.length >= 2)
      ) {
        Sentry.captureMessage(
          `Showed sync error dialog.\n${
            (JSON.stringify(serverSyncState.failures), null, 2)
          }`,
        )
        setShowSyncState(true)
      }
    }, [showSyncState, serverSyncState])

    if (rewindInfo && !simulationParamsForHistoryMode) {
      return (
        <div className="page h-screen flex flex-col justify-center items-center">
          <Spinner size="text-4xl" />
        </div>
      )
    }

    return (
      <>
        <WithSimulation
          params={
            rewindInfo
              ? fGet(simulationParamsForHistoryMode)
              : simulationParamsForPlanMode
          }
        />
        <CenteredModal
          className="dialog-outer-div"
          show={navGuardState.isTriggered && !showSyncState}
          onOutsideClickOrEscape={null}
        >
          {isSyncing ? (
            <>
              <h2 className=" dialog-heading">Still Syncing to Server</h2>
              <div className=" dialog-content-div relative h-[50px]">
                <Spinner size="text-4xl" />
              </div>
            </>
          ) : (
            <>
              <h2 className=" dialog-heading">Sync Completed</h2>
              <div className=" dialog-content-div  h-[50px] flex flex-col justify-center">
                <p className="p-base">
                  We prevented navigation because we were syncing your changes
                  to the server. Syncing is now complete. You can safely
                  navigate away from this page.
                </p>
              </div>
            </>
          )}
          <div className=" dialog-button-div">
            <button
              className=" dialog-button-dark"
              onClick={() => resetNavGuardState()}
            >
              Close
            </button>
          </div>
        </CenteredModal>
        <CenteredModal
          className="dialog-outer-div"
          show={showSyncState}
          onOutsideClickOrEscape={null}
        >
          <PlanServerImplSyncState
            syncState={serverSyncState}
            onHide={() => setShowSyncState(false)}
          />
        </CenteredModal>
      </>
    )
  },
)
PlanServerImpl.displayName = 'PlanServer'
