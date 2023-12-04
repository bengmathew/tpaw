import { assert, assertFalse, block, fGet, noCase } from '@tpaw/common'
import { clsx } from 'clsx'
import { formatDuration, intervalToDuration } from 'date-fns'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useEffect, useState } from 'react'
import { pluralize } from '../../../Utils/Pluralize'
import { useIANATimezoneName } from '../PlanRootHelpers/WithNonPlanParams'
import {
  SERVER_SYNC_PLAN_ERROR_WAIT_TIME,
  SERVER_SYNC_PLAN_THROTTLE_WAIT_TIME,
  ServerSyncState,
} from './UseServerSyncPlan'

type _SyncingOrErrorState = Extract<
  ServerSyncState,
  { type: 'syncing' } | { type: 'waitDueToError' }
>

export const PlanServerImplSyncState = React.memo(
  ({
    syncState,
    onHide,
  }: {
    syncState: ServerSyncState
    onHide: () => void
  }) => {
    const [dialogStartTime] = useState(Date.now())
    switch (syncState.type) {
      case 'synced':
      case 'waitDueToThrottle':
        return <_Success onHide={onHide} />
      case 'syncing':
      case 'waitDueToError': {
        const lastFailure = fGet(_.last(syncState.failures))
        const { showReload } = block(() => {
          switch (lastFailure.reason) {
            case 'timeout':
              return { showReload: true }
            case 'serverDownForMaintenance':
            case 'serverDownForUpdate':
              return { showReload: false }
            case 'other':
              return { showReload: true }
            default:
              noCase(lastFailure.reason)
          }
        })

        return (
          <div className="sm:px-2">
            <h2 className=" dialog-heading ">Error: Plan Not Saved</h2>
            <div className=" dialog-content-div">
              <div className="mb-4">
                <_Intro
                  className=""
                  dialogStartTime={dialogStartTime}
                  syncState={syncState}
                />
                <_Reason className="mt-4" reason={lastFailure.reason} />
                <_Retry className="mt-4" syncState={syncState} />
                {showReload && <_Reload className="mt-6" />}
              </div>
            </div>
          </div>
        )
      }
      default:
        noCase(syncState)
    }
  },
)
const _Success = React.memo(({ onHide }: { onHide: () => void }) => {
  return (
    <div className="">
      <h2 className=" dialog-heading">Success!</h2>
      <div className=" dialog-content-div">
        <p className="p-base">
          Your changes were saved to the server. You can continue updating your
          plan.
        </p>
      </div>
      <div className=" dialog-button-div">
        <button className=" dialog-button-dark" onClick={onHide}>
          Close
        </button>
      </div>
    </div>
  )
})

const _Intro = React.memo(
  ({
    className,
    dialogStartTime,
    syncState,
  }: {
    className?: string
    dialogStartTime: number
    syncState: _SyncingOrErrorState
  }) => {
    const { getZonedTime } = useIANATimezoneName()
    const issueStartTime =
      syncState.failures[0].timing.start - SERVER_SYNC_PLAN_THROTTLE_WAIT_TIME // Changes during a throttle might not have been saved.

    const issueStartTimeStr = getZonedTime(issueStartTime).toLocaleString(
      DateTime.TIME_WITH_SECONDS,
    )
    const dialogStartTimeStr = getZonedTime(dialogStartTime).toLocaleString(
      DateTime.TIME_WITH_SECONDS,
    )

    const timeSinceIssueStartStr = formatDuration(
      intervalToDuration({
        start: issueStartTime,
        end: dialogStartTime,
      }),
    )
    const numChanges =
      syncState.type === 'syncing'
        ? _getNumChangesInInput(syncState.inputSummary) +
          // This should always be null because we can't queue more changes once
          // this dialog shows up.
          (syncState.nextInputSummary ? assertFalse() : 0)
        : syncState.type === 'waitDueToError'
        ? _getNumChangesInInput(syncState.queuedSummary)
        : noCase(syncState)
    return (
      <p className={clsx(className, 'p-base')}>
        The last few changes that you made have not been saved to the server.
        This impacts the changes you made between {issueStartTimeStr} and{' '}
        {dialogStartTimeStr} (a duration of {timeSinceIssueStartStr}). You made
        approximately {pluralize(numChanges, 'change')} during this time.
      </p>
    )
  },
)

const _Reason = React.memo(
  ({
    className,
    reason,
  }: {
    className?: string
    reason: _SyncingOrErrorState['failures'][0]['reason']
  }) =>
    reason === 'other' ? (
      <></>
    ) : (
      <div className={clsx(className, '')}>
        <h2 className="font-bold mt-4">Reason</h2>
        {
          <p className="p-base mt-1">
            {reason === 'serverDownForMaintenance'
              ? `The server is down for maintenance. We expect the server to become available shortly. We will continue to retry until successful.`
              : reason === 'serverDownForUpdate'
              ? `The server is being updated to a new version. We expect the server to become available shortly. We will continue to retry until successful.`
              : reason === 'timeout'
              ? `The network call to the server timed out.`
              : noCase(reason)}
          </p>
        }
      </div>
    ),
)

const _Retry = React.memo(
  ({
    syncState,
    className,
  }: {
    syncState: _SyncingOrErrorState
    className?: string
  }) => {
    assert(syncState.failures.length > 0)
    const handleRetry =
      syncState.type === 'waitDueToError'
        ? syncState.retryNow
        : syncState.type === 'syncing'
        ? null
        : noCase(syncState)

    return (
      <div className={clsx(className, '')}>
        <h2 className="font-bold">Retry</h2>
        <p className="mt-1 p-base">
          We made {syncState.failures.length} unsuccessful attempts so far.
        </p>
        {syncState.type === 'waitDueToError' ? (
          syncState.waitEndTime === 'never' ? (
            <h2 className="">
              <span className="p-base">Retrying in: </span>
              <span className="lighten rounded-md bg-gray-200 px-2">
                Not automatically retrying
              </span>
            </h2>
          ) : (
            <div className="flex items-center gap-x-2">
              <h2 className="p-base">Retrying in:</h2>
              <_RetryBar
                className="mt-0.5"
                waitEndTime={syncState.waitEndTime}
              />
            </div>
          )
        ) : syncState.type === 'syncing' ? (
          <div className="flex items-center gap-x-2">
            <h2 className="p-base">Retrying now:</h2>
            <_RetryingBar className="mt-0.5" />
          </div>
        ) : (
          noCase(syncState)
        )}

        <button
          className="border border-gray-400  px-2 py-0 rounded-md disabled:lighten-2 mt-2"
          disabled={!handleRetry}
          onClick={handleRetry ?? undefined}
        >
          Retry Now
        </button>
      </div>
    )
  },
)
const _Reload = React.memo(({ className }: { className?: string }) => {
  return (
    <div className={clsx(className, 'border-t border-gray-300 pt-1')}>
      <div className="mt-2">
        {/* <h2 className="font-bold">Reload</h2> */}
        <p className="">
          <span className="p-base">
            If retrying does not work, you can try reloading the webpage. You
            will lose the unsaved changes.{' '}
          </span>
          <button
            className="border border-gray-400  px-2 py-0 rounded-md disabled:lighten-2 "
            onClick={() => window.location.reload()}
          >
            Reload and Lose Changes
          </button>
        </p>
      </div>
    </div>
  )
})

const _RetryBar = React.memo(
  ({ className, waitEndTime }: { className?: string; waitEndTime: number }) => {
    const [now, setNow] = useState(Date.now())
    React.useEffect(() => {
      const interval = window.setInterval(() => {
        setNow(Date.now())
      }, 100)
      return () => window.clearInterval(interval)
    }, [])

    const [startingPercent] = useState(
      (waitEndTime - now) / SERVER_SYNC_PLAN_ERROR_WAIT_TIME,
    )
    const [percent, setPercent] = useState(startingPercent)
    useEffect(() => {
      window.setTimeout(() => setPercent(0), 10)
    }, [])
    const width = 150

    return (
      <div className={clsx(className, 'flex items-center gap-x-2')}>
        <div
          className="relative bg-gray-200 rounded-full h-[5px] overflow-hidden"
          style={{ width: `${width}px` }}
        >
          <div
            className="absolute bg-gray-600 h-full w-full "
            style={{
              transitionProperty: 'transform',
              transitionTimingFunction: 'linear',
              transitionDuration: `${
                startingPercent * SERVER_SYNC_PLAN_ERROR_WAIT_TIME
              }ms`,
              transformOrigin: 'left',
              transform: `scaleX(${percent})`,
            }}
          />
        </div>
        <h2 className="">{((waitEndTime - now) / 1000).toFixed(0)} seconds </h2>
      </div>
    )
  },
)

const _RetryingBar = React.memo(({ className }: { className?: string }) => {
  const [side, setSide] = useState<0 | 1>(0)
  useEffect(() => {
    window.setTimeout(() => setSide(1), 10)
  }, [])
  const width = 150
  const inW = 20
  return (
    <div
      className={clsx(
        className,
        'relative bg-gray-200 rounded-full h-[5px] overflow-hidden',
      )}
      style={{ width: `${width}px` }}
    >
      <div
        className="absolute bg-gray-600 h-full "
        style={{
          width: `${inW}px`,
          transitionProperty: 'transform',
          animationIterationCount: 'infinite',
          transitionDuration: '1000ms',
          transform: `translateX(${side === 0 ? 0 : `${width - inW}px`})`,
        }}
        onTransitionEnd={() => setSide((x) => (x === 0 ? 1 : 0))}
      />
    </div>
  )
})

const _getNumChangesInInput = (
  inputSummary: Extract<ServerSyncState, { type: 'syncing' }>['inputSummary'],
) =>
  _.sumBy(inputSummary, (x) =>
    x.type === 'addItems'
      ? x.addCount
      : x.type === 'moveHead'
      ? 1
      : x.type === 'cutBranch'
      ? 1
      : noCase(x),
  )
