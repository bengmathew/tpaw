import _ from 'lodash'
import { DateTime } from 'luxon'
import { assert, fGet } from '../../Utils'

// This does not include the first item on the stack which cannot be undone.
export const MAX_UNDO_HISTORY_LENGTH = 100

export namespace PlanParamsHistoryFns {
  export const getUndoRedoStack = <T extends { id: string; index: number }>(
    history: T[],
    headIndex: number,
  ): { base: T[]; undo: T[]; redo: T[] } => {
    assert(history.length > 0)
    const undoStackStart = _.findLastIndex(
      history,
      (x, i) =>
        i === 0 ||
        history.length - i > MAX_UNDO_HISTORY_LENGTH ||
        x.index !== fGet(history[i - 1]).index + 1,
    )
    const undoRedoStackFlat = history.slice(undoStackStart)

    const headArrayIndex = undoRedoStackFlat.findIndex(
      (x) => x.index === headIndex,
    )
    assert(headArrayIndex !== -1)
    return {
      base: history.slice(0, undoStackStart),
      undo: undoRedoStackFlat.slice(0, headArrayIndex + 1),
      redo: undoRedoStackFlat.slice(headArrayIndex + 1),
    }
  }

  // Should satisfy.
  // filter([...preBase, ...postBase]) = filter(([...filter(preBase), ...postBase]))
  export const filterForLastChangePerDay = <
    HistoryItem extends { planParamsChangeId: string; timestamp: Date },
  >({
    ianaTimezoneName,
    planParamsHistory,
    intersectWithIds,
  }: {
    ianaTimezoneName: string
    planParamsHistory: HistoryItem[]
    intersectWithIds: Set<string> | null
  }) => {
    const withZonedTime = planParamsHistory.map((x) => ({
      ...x,
      zonedTime: DateTime.fromJSDate(x.timestamp, { zone: ianaTimezoneName }),
    }))
    const idsToDelete = new Set(
      withZonedTime
        .filter((x, i) => {
          if (i === 0 || i === planParamsHistory.length - 1) return false
          if (intersectWithIds && !intersectWithIds.has(x.planParamsChangeId))
            return false
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          return x.zonedTime.hasSame(withZonedTime[i + 1]!.zonedTime, 'day')
        })
        .map((x) => x.planParamsChangeId),
    )
    return { idsToDelete }
  }

  // Should satisfy.
  // filter([...preBase, ...postBase]) = filter(([...filter(preBase), ...postBase]))
  export const filterForHistoryFromStart = <
    HistoryItem extends { planParamsChangeId: string; timestamp: Date },
  >(data: {
    ianaTimezoneName: string
    marketCloses: number[] | 'useConservativeGuess'
    planParamsHistory: HistoryItem[]
  }) =>
    filterForLastChangePerDay({
      ...data,
      intersectWithIds: filterForCurrentPortfolioBalanceEstimation({
        ...data,
        endTime: 'untilEndOfData',
      }).idsToDelete,
    })

  // Should satisfy.
  // filter([...preBase, ...postBase]) = filter(([...filter(preBase), ...postBase]))
  export const filterForCurrentPortfolioBalanceEstimation = <
    HistoryItem extends { planParamsChangeId: string; timestamp: Date },
  >({
    endTime: endTimeIn,
    planParamsHistory,
    marketCloses: marketClosesIn,
  }: {
    endTime: number | 'untilEndOfData'
    marketCloses: number[] | 'useConservativeGuess'
    planParamsHistory: HistoryItem[]
  }) => {
    const startTime = fGet(_.first(planParamsHistory)).timestamp.getTime()
    const paramsEndTime = fGet(_.last(planParamsHistory)).timestamp.getTime()
    const marketCloses =
      marketClosesIn === 'useConservativeGuess'
        ? _getConservativeMarketCloses(
            startTime,
            DateTime.fromMillis(
              Math.max(
                // paramsEndTime may be ahead due to fastForward.
                startTime,
                Date.now(),
              ),
            )
              .plus({ days: 10 })
              .toMillis(),
          )
        : marketClosesIn
    const endTime =
      endTimeIn === 'untilEndOfData'
        ? marketCloses.length > 0
          ? Math.max(fGet(_.last(marketCloses)), paramsEndTime)
          : paramsEndTime
        : endTimeIn

    const withdrawalAndContributionTimes = _getWithdrawalAndContributionTimes(
      startTime,
      endTime,
    )

    const monthlyRebalanceTimes = _getMonthlyRebalanceTimes(startTime, endTime)

    type Action =
      | { isParam: false; timestamp: number }
      | { isParam: true; id: string; timestamp: number }
    const combined = combineActionsForPortfolioBalanceEstimation<Action>(
      {
        withdrawalAndContributionActions: withdrawalAndContributionTimes.map(
          (x) => ({
            isParam: false,
            timestamp: x,
          }),
        ),
        monthlyRebalanceActions: monthlyRebalanceTimes.map((x) => ({
          isParam: false,
          timestamp: x,
        })),
        marketCloseActions: marketCloses.map((x) => ({
          isParam: false,
          timestamp: x,
        })),
        planChangeActions: planParamsHistory.map((x) => ({
          isParam: true,
          id: x.planParamsChangeId,
          timestamp: x.timestamp.getTime(),
        })),
      },
      (x) => x.timestamp,
    )

    const firstParamsIndex = combined.findIndex((x) => x.isParam)

    const idsToDelete = new Set<string>()
    combined.forEach((x, i) => {
      if (i !== firstParamsIndex && x.isParam && combined[i + 1]?.isParam) {
        idsToDelete.add(x.id)
      }
    })

    return {
      startTime,
      withdrawalAndContributionTimes,
      monthlyRebalanceTimes,
      idsToDelete,
    }
  }


// Note: This is mirrored in packages/simulator-rust.
  export const combineActionsForPortfolioBalanceEstimation = <T>(
    x: {
      marketCloseActions: T[]
      monthlyRebalanceActions: T[]
      planChangeActions: T[]
      withdrawalAndContributionActions: T[]
    },
    getTime: (t: T) => number,
  ) => {
    // Stable sort, so this order reflects action priority for identical
    // timestamp.
    return [
      // plan change has to be first because it is implicitly first for the
      // first plan change because that is interpreted as the start state and
      // removed before generating the rest of the actions. If we didn't keep
      // this consistency with the rest of the plan changes, we won't be able to
      // merge. For merging we have to line up the states for last plan change
      // on the left and the first plan change on the right, which will not be
      // possible if they generate actions in the wrong order for the first and
      // last plan changes.
      ...x.planChangeActions,
      ...x.marketCloseActions,
      ...x.monthlyRebalanceActions,
      ...x.withdrawalAndContributionActions,
    ].sort((a, b) => getTime(a) - getTime(b))
  }
}

// start is closed, end is closed.
const _getStartOfMonths = (start: DateTime, end: DateTime) => {
  let first = start.startOf('month')
  if (first !== start) first = first.plus({ month: 1 })
  const last = end.startOf('month')
  const numMonths = fGet(last.diff(first, 'month').toObject().months) + 1
  assert(_.isInteger(numMonths))
  return _.times(numMonths).map((i) => first.plus({ month: i }))
}

const _getWithdrawalAndContributionTimes = (start: number, end: number) => {
  const inUTC = (x: number) => DateTime.fromMillis(x).setZone('utc')
  return _getStartOfMonths(inUTC(start), inUTC(end).plus({ hours: 25 }))
    .map((x) => x.minus({ hours: 24 }).toMillis())
    .filter((x) => x >= start && x <= end)
}

const _getMonthlyRebalanceTimes = (start: number, end: number) => {
  const inUTC = (x: number) => DateTime.fromMillis(x).setZone('utc')
  return _getStartOfMonths(inUTC(start).minus({ hours: 25 }), inUTC(end))
    .map((x) => x.plus({ hours: 24 }).toMillis())
    .filter((x) => x >= start && x <= end)
}

// start is closed, end in closed.
const _getConservativeMarketCloses = (start: number, end: number) => {
  const startNY = DateTime.fromMillis(start)
    .setZone('America/New_York')
    .minus({ days: 1 })
    .set({ hour: 16 })
  const endNY = DateTime.fromMillis(end)
    .setZone('America/New_York')
    .plus({ days: 1 })
    .set({ hour: 16 })

  const numDays = Math.ceil(
    fGet(endNY.diff(startNY, 'days').toObject().days) + 1,
  )
  return _.times(numDays)
    .map((i) => startNY.plus({ day: i }))
    .filter((x) => !(x.weekdayShort === 'Sun' || x.weekdayShort === 'Sat'))
    .map((x) => x.toMillis())
}
