import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faTurnDownLeft } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, assertFalse, fGet, noCase } from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { ParamsExtended } from '../../../TPAWSimulator/ExtentParams'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../Utils/Geometry'
import { groupBy } from '../../../Utils/GroupBy'
import {
  CurrentPortfolioBalanceEstimateAction,
  CurrentPortfolioBalanceEstimateInfo,
  estimateCurrentPortfolioBalance,
} from '../../App/EstimateCurrentPortfolioBalance'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../Common/Inputs/SmartDeltaFnForAmountInput'
import { ModalBase } from '../../Common/Modal/ModalBase'
import { planSectionLabel } from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputCurrentPortfolioBalance = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {
      currentPortfolioBalanceEstimate,
      setCurrentPortfolioBalance,
      paramsExt,
    } = useSimulation()
    const { getDateTimeInCurrentTimezone } = paramsExt

    const isEstimate = useMemo(
      () => _isReallyEstimate(currentPortfolioBalanceEstimate),
      [currentPortfolioBalanceEstimate],
    )

    const [showExplanation, setShowExplanation] = useState(false)

    const formatTime = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('LLLL d, yyyy')

    const handleChange = (amount: number) => {
      if (amount === currentPortfolioBalanceEstimate.value) return
      setCurrentPortfolioBalance(Math.max(amount, 0))
    }
    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="params-card"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          >
            <p className="p-base">
              {`Enter the total value of the savings that you have accumulated so
              far. Include only investment accounts. Don't include home equity
              or the value of real estate holdings.`}
            </p>
            <div className="mt-4 flex">
              <AmountInput
                className="text-input"
                prefix="$"
                value={currentPortfolioBalanceEstimate.value}
                onChange={handleChange}
                decimals={0}
                modalLabel={planSectionLabel('current-portfolio-balance')}
              />
              <button
                className="ml-2 px-3"
                onClick={() => {
                  handleChange(
                    smartDeltaFnForAmountInput.increment(
                      currentPortfolioBalanceEstimate.value,
                    ),
                  )
                }}
              >
                <FontAwesomeIcon icon={faPlus} />
              </button>
              <button
                className="px-3"
                onClick={() => {
                  handleChange(
                    smartDeltaFnForAmountInput.decrement(
                      currentPortfolioBalanceEstimate.value,
                    ),
                  )
                }}
              >
                <FontAwesomeIcon icon={faMinus} />
              </button>
            </div>
            {isEstimate && (
              <div className="mt-4 ml-2">
                <h2 className="flex items-top gap-x-1">
                  {' '}
                  <FontAwesomeIcon
                    className="rotate-90 mr-1"
                    icon={faTurnDownLeft}
                  />
                  <div className="">
                    <h2>
                      This is an estimate calculated from your last entry of{' '}
                      {formatCurrency(
                        currentPortfolioBalanceEstimate.entered.amount,
                      )}{' '}
                      on{' '}
                      {formatTime(
                        currentPortfolioBalanceEstimate.entered.timestamp,
                      )}
                      .{' '}
                      <button
                        className="underline block"
                        onClick={() => setShowExplanation(true)}
                      >
                        View estimation steps
                      </button>
                    </h2>
                  </div>
                </h2>
                {showExplanation && (
                  <ModalBase
                    onClose={() => setShowExplanation(false)}
                    maxHeight="90vh"
                  >
                    {() => (
                      <_EstimatePopUp
                        currentPortfolioBalanceEstimate={
                          currentPortfolioBalanceEstimate
                        }
                      />
                    )}
                  </ModalBase>
                )}
              </div>
            )}
          </div>
        </div>
      </PlanInputBody>
    )
  },
)

type _ProcessedActions = ReturnType<typeof _processActions>
const _processActions = (
  { actions, entered }: CurrentPortfolioBalanceEstimateInfo,
  paramsExt: ParamsExtended,
) => {
  const { getDateTimeInCurrentTimezone, currentTime } = paramsExt
  const monthBoundaries = new Map(
    [
      ...groupBy(
        // This modifies actions.
        _.filter(actions, (x) => x.args.type === 'monthBoundary'),
        (x) => x.time,
      ).entries(),
    ].map(([time, value]) => {
      assert(value.length === 1)
      const { args, stateChange } = value[0]
      assert(args.type === 'monthBoundary')
      return [time, { args, stateChange }]
    }),
  )
  actions = [...actions].reverse()
  const byMonth = _.sortBy(
    [
      ...groupBy(actions, (action) =>
        getDateTimeInCurrentTimezone
          .fromMillis(action.time)
          .startOf('month')
          .valueOf(),
      ).entries(),
    ],
    ([month]) => -month,
  )

  const result = byMonth.map(([month, actions], monthIndex) => {
    const startingState = (() => {
      const firstAction = fGet(_.last(actions))
      const allocation = firstAction.stateChange.start.allocation
      const estimate =
        firstAction.args.type === 'monthBoundary'
          ? firstAction.stateChange.end.estimate
          : firstAction.stateChange.start.estimate
      return { allocation, estimate }
    })()
    actions = actions.filter((x) => x.args.type !== 'monthBoundary')
    const nextMonthBoundary = monthBoundaries.get(
      getDateTimeInCurrentTimezone
        .fromMillis(month)
        .plus({ month: 1 })
        .valueOf(),
    )
    if (!nextMonthBoundary) {
      // Has to be the last month, or sometimes the first month. If it is the
      // first month, we filter it later. Nothing really happens.
      assert(monthIndex === 0 || monthIndex === byMonth.length - 1)
    }
    const endOfMonth = nextMonthBoundary
      ? {
          contributionOrWithdrawal:
            nextMonthBoundary.args.netContributionOrWithdrawal,
          stateChange: nextMonthBoundary.stateChange,
        }
      : null

    const monthBoundary = monthBoundaries.get(month)
    const startOfMonth = (() => {
      if (monthIndex === byMonth.length - 1) {
        // First month.
        assert(!monthBoundary)
        assert(actions[actions.length - 1].args.type === 'planChangeRebalance')
        actions = _.take(actions, actions.length - 1)
        return null
      } else {
        if (!monthBoundary) {
          // Sometimes second month may not have a boundary. This can happen
          // if last entry was in this month but last market close was in
          // previous month, so even though there is an entry for last month
          // we don't cross the boundary for withdrawals and reallocations.
          assert(monthIndex === byMonth.length - 2)
          return null
        } else {
          return {
            allocation: monthBoundary.args.allocation,
            stateChange: monthBoundary.stateChange,
          }
        }
      }
    })()
    const byDay = groupBy(actions, (action) =>
      getDateTimeInCurrentTimezone
        .fromMillis(action.time)
        .startOf('day')
        .valueOf(),
    )
    const allDays = [getDateTimeInCurrentTimezone.fromMillis(month)]
    while (allDays[allDays.length - 1].month === allDays[0].month) {
      allDays.push(allDays[allDays.length - 1].plus({ day: 1 }))
    }
    allDays.pop()

    allDays
      .map((x) => x.valueOf())
      .filter((x) => x <= currentTime.valueOf() && x >= entered.timestamp)
      .forEach((day) => {
        if (!byDay.has(day)) byDay.set(day, [])
      })
    return {
      month,
      startingState,
      startOfMonth,
      endOfMonth,
      byDay: _.sortBy(
        [...byDay.entries()].map(([day, actions]) => ({
          day,
          actions,
        })),
        (x) => -x.day,
      ),
    }
  })
  const firstMonth = fGet(_.last(result))
  assert(!firstMonth.startOfMonth)
  // Nothing really happened this month.
  if (firstMonth.byDay.length === 0 && firstMonth.endOfMonth === null)
    result.pop()
  return result
}

const _EstimatePopUp = React.memo(
  ({
    currentPortfolioBalanceEstimate,
  }: {
    currentPortfolioBalanceEstimate: ReturnType<
      typeof estimateCurrentPortfolioBalance
    >
  }) => {
    const { paramsExt, currentTime } = useSimulation()
    const { entered } = currentPortfolioBalanceEstimate
    const { getDateTimeInCurrentTimezone } = paramsExt
    const byMonth = useMemo(
      () => _processActions(currentPortfolioBalanceEstimate, paramsExt),
      [currentPortfolioBalanceEstimate, paramsExt],
    )

    const formatTime = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('LLLL d, yyyy')
    return (
      <div className="p-4 min-w-[600px] sm:min-w-auto">
        {/* <h2 className="text-right">Zone: {currentTime.toFormat('ZZZZ')}</h2> */}
        <div className="ml-4">
          <h2 className="font-bold text-lg">You Entered</h2>
          <div className="">
            <h2>
              {' '}
              {formatCurrency(entered.amount)} on{' '}
              {formatTime(entered.timestamp)}
            </h2>
          </div>
          <h2 className="font-bold mt-2 text-lg">Current Estimate</h2>
          <div className="">
            <h2>
              {' '}
              <span className="">
                {formatCurrency(currentPortfolioBalanceEstimate.value)}
              </span>{' '}
              as of {formatTime(currentTime.valueOf())}
            </h2>
          </div>
          <h2 className="mt-2">This estimate assumes you are invested in:</h2>
          <div className="">
            <h2>
              Stocks{' — '}
              <span className="">Vanguard Total World Stock ETF (VT)</span>
            </h2>
            <h2>Bonds — Vanguard Total Bond Market ETF (BND)</h2>
          </div>
        </div>
        {/* <h2 className="font-bold mt-4">Changes</h2> */}

        {byMonth.length === 0 ? (
          <h2 className="">
            None. No market closes or month boundary occurred since the last
            plan update.
          </h2>
        ) : (
          byMonth.map((info, i) => (
            <_Month
              key={i}
              className="mt-4 mb-10 rounded-lg  bg-gray-100 px-4 pb-4"
              info={info}
            />
          ))
        )}
      </div>
    )
  },
)

const _Month = React.memo(
  ({
    className = '',
    info: { month, byDay, endOfMonth, startOfMonth, startingState },
  }: {
    className?: string
    info: _ProcessedActions[0]
  }) => {
    const { paramsExt } = useSimulation()
    const { currentTime, getDateTimeInCurrentTimezone } = paramsExt
    const monthDateTime = useMemo(
      () => getDateTimeInCurrentTimezone.fromMillis(month),
      [getDateTimeInCurrentTimezone, month],
    )
    return (
      <div className={`${className} text-sm`}>
        <h2 className="inline-block font-bold text-2xl sticky -top-2 sm:-top-4 bg-gray-100 pt-4 pr-4 pb-2 rounded-br-lg z-10">
          <span className="pt-4 ">{monthDateTime.toFormat('LLLL yyyy')}</span>
        </h2>
        {endOfMonth && (
          <div className="mt-2 ">
            <h2 className="font-bold text-base">At End of Month</h2>
            <h2 className="mt-1 ">
              {endOfMonth.contributionOrWithdrawal.type === 'contribution' ? (
                <>
                  Contributed{' '}
                  {formatCurrency(
                    endOfMonth.contributionOrWithdrawal.contribution,
                  )}{' '}
                </>
              ) : (
                <>
                  Withdrew{' '}
                  {formatCurrency(
                    endOfMonth.contributionOrWithdrawal.withdrawal,
                  )}{' '}
                </>
              )}
              leaving the balance at{' '}
              {formatCurrency(endOfMonth.stateChange.end.estimate)}.
            </h2>
          </div>
        )}
        {byDay.length > 0 && (
          <>
            {' '}
            <h2 className="font-bold mt-4 text-base">Daily Activity</h2>
            <div
              className="inline-grid items-center gap-x-3 mt-1"
              style={{
                grid: 'auto / auto auto auto auto auto auto auto ',
              }}
            >
              {/* <h2 className="col-span-2 text-center">Day</h2>
          <h2 className="">Label</h2>
          <h2 className=" col-span-3">Amount</h2>
          <h2 className=" col-span-3">Allocation</h2> */}
              {byDay.map((info, i) => (
                <_Day key={i} info={info} />
              ))}
            </div>
          </>
        )}
        {/* {startOfMonth && ( */}
        <div className="mt-4">
          {startOfMonth ? (
            <>
              <h2 className="font-bold text-base">At Start of Month</h2>{' '}
              <h2 className="mt-1">
                Rebalanced {formatCurrency(startingState.estimate)} to{' '}
                {formatPercentage(0)(startingState.allocation.stocks)} stocks.
              </h2>
            </>
          ) : (
            <>
              <h2 className="font-bold text-base">Started With</h2>{' '}
              <h2 className="mt-1">
                {formatCurrency(startingState.estimate)} with{' '}
                {formatPercentage(0)(startingState.allocation.stocks)} in
                stocks.
              </h2>
            </>
          )}
        </div>
        {/* )} */}
      </div>
    )
  },
)

const _Day = React.memo(
  ({
    info: { day, actions },
  }: {
    info: { day: number; actions: CurrentPortfolioBalanceEstimateAction[] }
  }) => {
    const { paramsExt } = useSimulation()
    const { getDateTimeInCurrentTimezone } = paramsExt
    const formatMonth = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('LLL')
    const formatDayOfWeek = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('EEE')
    const formatDay = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('d')
    return (
      <>
        {actions.length === 0 ? (
          <>
            <h2 className="font-medium text-left lighten-2">
              {formatDayOfWeek(day)}
            </h2>
            <h2 className="font-medium text-right lighten-2">
              {formatMonth(day)}
            </h2>
            <h2 className="font-medium text-center lighten-2">
              {formatDay(day)}
            </h2>
            <h2 className="font-medium col-span-4 lighten-2 text-sm ">— </h2>
          </>
        ) : (
          actions.map((action, i) => (
            <_Action key={i} action={action} isLastActionForDay={i === 0} />
          ))
        )}
      </>
    )
  },
)

const _Action = React.memo(
  ({
    action,
    isLastActionForDay,
  }: {
    action: CurrentPortfolioBalanceEstimateAction
    isLastActionForDay: boolean
  }) => {
    const { paramsExt } = useSimulation()
    const { getDateTimeInCurrentTimezone } = paramsExt
    const {
      time,
      stateChange: { start, end },
      args,
    } = action
    const formatMonth = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('LLL')

    const formatDayOfWeek = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('EEE')
    const formatDay = (x: number) =>
      getDateTimeInCurrentTimezone.fromMillis(x).toFormat('d')

    const formatStock = (x: number) => {
      const sign = x < 0 ? '-' : x === 0 ? '' : '+'
      return `${sign}${formatPercentage(2)(Math.abs(x))}`
    }

    const color = (x: number) => (x >= 0 ? 'text-green-500' : 'text-red-400')

    return (
      <>
        {isLastActionForDay ? (
          <>
            <h2 className="font-medium text-left">{formatDayOfWeek(time)}</h2>
            <h2 className="font-medium text-right">{formatMonth(time)}</h2>
            <h2 className="font-medium text-center">{formatDay(time)}</h2>
          </>
        ) : (
          <h2 className="col-span-3"></h2>
        )}
        {(() => {
          switch (args.type) {
            case 'marketClose':
              return (
                <div
                  className="grid items-center text-right gap-x-2 "
                  style={{ grid: 'auto/ 15px 45px 30px 45px' }}
                >
                  <h2 className="text-xs">VT</h2>
                  <h2
                    className={`text-right ${color(
                      args.marketData.percentageChangeFromLastClose.vt,
                    )}`}
                  >
                    {formatStock(
                      args.marketData.percentageChangeFromLastClose.vt,
                    )}
                  </h2>
                  <h2 className="text-xs">BND</h2>
                  <h2
                    className={`text-right ${color(
                      args.marketData.percentageChangeFromLastClose.bnd,
                    )}`}
                  >
                    {formatStock(
                      args.marketData.percentageChangeFromLastClose.bnd,
                    )}
                  </h2>
                </div>
              )
            case 'planChangeRebalance':
              return <h2 className="">Rebalance </h2>
            case 'monthBoundary':
              assertFalse()
            default:
              noCase(args)
          }
        })()}
        {/* <h2 className="">{formatCurrency(start.estimate)}</h2>
        <h2 className="text-xs">
          <FontAwesomeIcon icon={faRightLong} />
        </h2> */}
        <h2 className="">
          {formatPercentage(0)(end.allocation.stocks)} stocks
        </h2>
        <h2 className="text-right">{formatCurrency(end.estimate)}</h2>
        <h2 className={`text-right ${color(end.estimate - start.estimate)}`}>
          {formatCurrency(end.estimate - start.estimate)}
        </h2>
      </>
    )
  },
)

export const CurrentPortfolioBalanceSummary = React.memo(() => {
  const { currentPortfolioBalanceEstimate, paramsExt } = useSimulation()
  const { value, entered } = currentPortfolioBalanceEstimate
  const isEstimate = useMemo(
    () => _isReallyEstimate(currentPortfolioBalanceEstimate),
    [currentPortfolioBalanceEstimate],
  )
  const { getDateTimeInCurrentTimezone } = paramsExt

  const formatTime = (x: number) =>
    getDateTimeInCurrentTimezone.fromMillis(x).toFormat('LLLL d, yyyy')

  return (
    <>
      <h2>{formatCurrency(value)}</h2>
      {isEstimate && (
        <>
          <h2>
            Estimated from your last entry of {formatCurrency(entered.amount)}{' '}
            on {formatTime(entered.timestamp)}
          </h2>
          {/* <h2 className="ml-4">
            {formatCurrency(entered.amount)} on {formatTime(entered.timestamp)}
          </h2> */}
        </>
        // <h2>
        //   Estimated from your entry of{' '}
        //   {formatCurrency(entered.amount)} on {formatTime(entered.timestamp)}.{' '}
        // </h2>
      )}
    </>
  )
})

const _isReallyEstimate = (
  currentPortfolioBalanceEstimate: ReturnType<
    typeof estimateCurrentPortfolioBalance
  >,
) =>
  currentPortfolioBalanceEstimate.actions.some((x) => {
    switch (x.args.type) {
      case 'marketClose':
      case 'monthBoundary':
        return true
      case 'planChangeRebalance':
        return false
      default:
        noCase(x.args)
    }
  })
