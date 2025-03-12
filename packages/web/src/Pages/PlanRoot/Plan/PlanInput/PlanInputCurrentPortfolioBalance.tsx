import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faTurnDownLeft } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, fGet, getZonedTimeFns, letIn, noCase } from '@tpaw/common'
import { default as clix, default as clsx } from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useEffect, useMemo, useState } from 'react'
import { PlanParamsNormalized } from '@tpaw/common'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../Utils/Geometry'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../PlanRootHelpers/WithSimulation'
import { planSectionLabel } from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'
import { PortfolioBalanceEstimation } from '../../../../Simulator/SimulateOnServer/SimulateOnServer'
import { groupBy } from '../../../../Utils/GroupBy'

export const PlanInputCurrentPortfolioBalance = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <_CurrentPortfolioBalanceCard props={props} />
      </PlanInputBody>
    )
  },
)

export const _CurrentPortfolioBalanceCard = React.memo(
  ({
    className,
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const [showHistory, setShowHistory] = useState(false)
    const { updatePlanParams } = useSimulationInfo()
    const { simulationResult } = useSimulationResultInfo()
    const estimateInfo = useEstimateInfo()

    // Need to keep a local copy of amount and not use directly from
    // simulationResult because the update rate is far too slow.
    const [amount, setAmount] = useState('fromResult' as 'fromResult' | number)

    useEffect(() => {
      setAmount(
        simulationResult.portfolioBalanceEstimationByDated.currentBalance,
      )
    }, [simulationResult])

    const handleChange = (amount: number) => {
      setAmount(amount)
      updatePlanParams('setCurrentPortfolioBalance', amount)
    }

    return (
      <div className="">
        <div
          className={clix(className, 'params-card')}
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <p className="p-base">
            {`Enter the total value of the savings that you have accumulated so
        far. Include only investment accounts. Don't include home equity
        or the value of real estate holdings.`}
          </p>
          <div className="mt-4 flex">
            {letIn(
              amount === 'fromResult'
                ? simulationResult.portfolioBalanceEstimationByDated
                    .currentBalance
                : amount,
              (amount) => (
                <>
                  <AmountInput
                    className="text-input"
                    prefix="$"
                    value={amount}
                    onChange={(x) => {
                      // This happens on blur and enter. The check for real change is
                      // especially important when amount is an estimate when
                      // handleChange() is called, we mark it as not a estimate, which
                      // is materially different. We don't want to do that on blur and
                      // enter if the value has not changed.
                      if (x !== amount) handleChange(x)
                    }}
                    decimals={0}
                    modalLabel={planSectionLabel('current-portfolio-balance')}
                  />
                  <button
                    className="ml-2 px-3"
                    onClick={() => {
                      handleChange(smartDeltaFnForAmountInput.increment(amount))
                    }}
                  >
                    <FontAwesomeIcon icon={faPlus} />
                  </button>
                  <button
                    className="px-3"
                    onClick={() => {
                      handleChange(smartDeltaFnForAmountInput.decrement(amount))
                    }}
                  >
                    <FontAwesomeIcon icon={faMinus} />
                  </button>
                </>
              ),
            )}
          </div>
          {estimateInfo.type === 'notAnEstimate' ? (
            <></>
          ) : estimateInfo.type === 'estimate' ? (
            <div className={'mt-4 ml-2 flex items-top gap-x-1'}>
              <FontAwesomeIcon
                className="rotate-90 mr-1"
                icon={faTurnDownLeft}
              />
              <div className="">
                <p className="p-base">
                  This is an estimate calculated from your last entry of{' '}
                  {formatCurrency(estimateInfo.lastEnteredAmount)} on{' '}
                  {getZonedTimeFns(estimateInfo.ianaTimezoneName)(
                    estimateInfo.lastEnteredTimestamp,
                  ).toLocaleString(DateTime.DATE_MED)}
                  .{' '}
                </p>
              </div>
            </div>
          ) : (
            noCase(estimateInfo)
          )}
          {simulationResult.portfolioBalanceEstimationByDated.isDatedPlan && (
            <>
              <button
                className="underline block mt-4"
                onClick={() => setShowHistory(true)}
              >
                View Balance History
              </button>
              <CenteredModal
                className=" dialog-outer-div"
                show={showHistory}
                onOutsideClickOrEscape={() => setShowHistory(false)}
              >
                <_Popup
                  detail={
                    simulationResult.portfolioBalanceEstimationByDated.detail
                  }
                  planParamsNormOfResult={
                    simulationResult.planParamsNormOfResult
                  }
                />
              </CenteredModal>
            </>
          )}
        </div>
      </div>
    )
  },
)

const _Popup = React.memo(
  ({
    detail,
    planParamsNormOfResult,
  }: {
    detail: PortfolioBalanceEstimation.Detail
    planParamsNormOfResult: PlanParamsNormalized
  }) => {
    const { ianaTimezoneName, getZonedTime } = useIANATimezoneName()
    const byMonthDetail = useMemo(
      () =>
        getByMonthDetail(detail, ianaTimezoneName),
      [detail, ianaTimezoneName],
    )
    const { datingInfo } = planParamsNormOfResult
    assert(datingInfo.isDated)
    const currentTimestamp = datingInfo.nowAsTimestamp
    const currentEstimate =
      _.last(detail.actions)?.stateChange.end.estimate ??
      detail.startState.estimate

    const formatTime = (x: number) => getZonedTime(x).toFormat('LLLL d, yyyy')
    return (
      <div className="p-2 min-w-[600px] sm:min-w-auto">
        <div className="ml-4">
          <h2 className="font-bold mt-2 text-lg">Current Estimate</h2>
          <div className="">
            <h2>
              {' '}
              <span className="">{formatCurrency(currentEstimate)}</span> as of{' '}
              {formatTime(currentTimestamp)}
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

        {byMonthDetail.monthsDesc.map((info) => (
          <_Month
            key={`month-${info.month}`}
            className="mt-4 mb-10 rounded-lg  bg-gray-100 px-4 pb-4"
            info={info}
          />
        ))}
      </div>
    )
  },
)

type MonthInfo = ByMonthInfo['monthsDesc'][0]
const _Month = React.memo(
  ({
    className = '',
    info: { month, stateChange, daysDesc },
  }: {
    className?: string
    info: MonthInfo
  }) => {
    const { getZonedTime } = useIANATimezoneName()
    return (
      <div className={`${className} text-sm`}>
        <h2 className="inline-block font-bold text-2xl sticky -top-2 sm:-top-4 bg-gray-100 pt-4 pr-4 pb-2 rounded-br-lg z-10">
          <span className="pt-4 ">
            {getZonedTime(month).toFormat('LLLL yyyy')}
          </span>
        </h2>
        <div className="mt-4">
          <h2 className="font-bold text-base">Ended With</h2>{' '}
          <h2 className="mt-1">
            {formatCurrency(stateChange.end.estimate)} with{' '}
            {formatPercentage(0)(stateChange.end.allocation.stocks)} in stocks.
          </h2>
        </div>
        <>
          <h2 className="font-bold mt-4 text-base">Daily Activity</h2>
          <div
            className="inline-grid items-center gap-x-3 mt-1"
            style={{
              grid: 'auto / auto auto auto auto auto auto auto ',
            }}
          >
            {daysDesc.map((info) => (
              <_Day key={info.day} info={info} />
            ))}
          </div>
        </>
        <div className="mt-4">
          <h2 className="font-bold text-base">Started With</h2>{' '}
          <h2 className="mt-1">
            {formatCurrency(stateChange.start.estimate)} with{' '}
            {formatPercentage(0)(stateChange.start.allocation.stocks)} in
            stocks.
          </h2>
        </div>
      </div>
    )
  },
)

const _Day = React.memo(
  ({ info: { day, actionsDesc } }: { info: MonthInfo['daysDesc'][number] }) => {
    const { ianaTimezoneName, getZonedTime } = useIANATimezoneName()
    const formatMonth = (x: DateTime) => x.toFormat('LLL')
    const formatDayOfWeek = (x: DateTime) => x.toFormat('EEE')
    const formatDay = (x: DateTime) => x.toFormat('d')
    return (
      <>
        {actionsDesc.length === 0 ? (
          <>
            <h2 className="font-medium text-left lighten-2">
              {formatDayOfWeek(getZonedTime(day))}
            </h2>
            <h2 className="font-medium text-right lighten-2">
              {formatMonth(getZonedTime(day))}
            </h2>
            <h2 className="font-medium text-center lighten-2">
              {formatDay(getZonedTime(day))}
            </h2>
            <h2 className="font-medium col-span-4 lighten-2 text-sm ">— </h2>
          </>
        ) : (
          actionsDesc.map((action, i) => (
            <_Action
              key={i}
              action={action}
              isLastActionForDay={i === 0}
              ianaTimezoneName={ianaTimezoneName}
            />
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
    ianaTimezoneName,
  }: {
    action: PortfolioBalanceEstimation.Action
    isLastActionForDay: boolean
    ianaTimezoneName: string
  }) => {
    const {
      timestamp,
      stateChange: { start, end },
      args,
    } = action
    const formatted = useMemo(() => {
      const dateTime = DateTime.fromMillis(timestamp).setZone(ianaTimezoneName)
      const month = dateTime.toFormat('LLL')
      const dayOfWeek = dateTime.toFormat('EEE')
      const day = dateTime.toFormat('d')
      return { dateTime, month, dayOfWeek, day }
    }, [ianaTimezoneName, timestamp])

    const formatStock = (x: number) => {
      const sign = x < 0 ? '-' : x === 0 ? '' : '+'
      return `${sign}${formatPercentage(2)(Math.abs(x))}`
    }

    const color = (x: number) => (x >= 0 ? 'text-green-500' : 'text-red-400')

    return (
      <>
        {isLastActionForDay ? (
          <>
            <h2 className="font-medium text-left">{formatted.dayOfWeek}</h2>
            <h2 className="font-medium text-right">{formatted.month}</h2>
            <h2 className="font-medium text-center">{formatted.day}</h2>
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
            case 'planChange':
              return <h2 className="">Plan Updated </h2>
            case 'withdrawalAndContribution':
              return (
                <h2 className="">
                  {args.netContributionOrWithdrawal.type === 'contribution'
                    ? 'Contribution'
                    : 'Withdrawal'}
                </h2>
              )
            case 'monthlyRebalance':
              return <h2 className="">Monthly Rebalance </h2>
            default:
              noCase(args)
          }
        })()}
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

export const PlanInputCurrentPortfolioBalanceSummary = React.memo(
  ({ forPrint }: { forPrint: boolean }) => {
    const estimateInfo = useEstimateInfo()
    const { simulationResult } = useSimulationResultInfo()
    return (
      <>
        <h2>
          {formatCurrency(
            simulationResult.portfolioBalanceEstimationByDated.currentBalance,
          )}
        </h2>
        {estimateInfo.type === 'notAnEstimate' ? (
          <></>
        ) : estimateInfo.type === 'estimate' ? (
          <div className={'ml-2 flex items-top gap-x-1'}>
            <FontAwesomeIcon
              className={clsx('rotate-90 mr-1', forPrint && 'text-[10px]')}
              icon={faTurnDownLeft}
            />
            <div className="">
              <p className={clsx(forPrint ? '' : 'font-font2 text-base')}>
                This is an estimate calculated from your last entry of{' '}
                {formatCurrency(estimateInfo.lastEnteredAmount)} on{' '}
                {getZonedTimeFns(estimateInfo.ianaTimezoneName)(
                  estimateInfo.lastEnteredTimestamp,
                ).toLocaleString(DateTime.DATE_MED)}
                .{' '}
              </p>
            </div>
          </div>
        ) : (
          noCase(estimateInfo)
        )}
      </>
    )
  },
)

type EstimateInfo =
  | { type: 'notAnEstimate' }
  | {
      type: 'estimate'
      lastEnteredAmount: number
      lastEnteredTimestamp: number
      ianaTimezoneName: string
    }
const useEstimateInfo = (): EstimateInfo => {
  const { simulationResult } = useSimulationResultInfo()
  return useMemo((): EstimateInfo => {
    const { portfolioBalanceEstimationByDated } = simulationResult
    if (!portfolioBalanceEstimationByDated.isDatedPlan)
      return { type: 'notAnEstimate' }
    const { detail } = portfolioBalanceEstimationByDated
    if (detail.actions.length === 0) return { type: 'notAnEstimate' }
    

    const lastPortfolioUpdateIndex = _.findLastIndex(
      detail.actions,
      (action) =>
        action.args.type === 'planChange' &&
        action.args.portfolioUpdate !== null,
    )
    if (lastPortfolioUpdateIndex === detail.actions.length - 1)
      return { type: 'notAnEstimate' }
    const ianaTimezoneName = fGet(
      simulationResult.ianaTimezoneNameIfDatedPlanOfResult,
    )
    const result: EstimateInfo =
      lastPortfolioUpdateIndex === -1
        ? {
            type: 'estimate',
            lastEnteredAmount: detail.startState.estimate,
            lastEnteredTimestamp: detail.startTimestamp,
            ianaTimezoneName,
          }
        : letIn(detail.actions[lastPortfolioUpdateIndex], ({ args }) => {
            assert(args.type === 'planChange' && args.portfolioUpdate)
            return {
              type: 'estimate',
              lastEnteredAmount: args.portfolioUpdate.amount,
              lastEnteredTimestamp: args.portfolioUpdate.exactTimestamp,
              ianaTimezoneName,
            }
          })
    if (
      result.lastEnteredAmount ==
      simulationResult.portfolioBalanceEstimationByDated.currentBalance
    )
      return { type: 'notAnEstimate' }
    return result
  }, [simulationResult])
}



export type ByMonthInfo = ReturnType<typeof getByMonthDetail>

// Note, this can be empty since info.actions can be empty.
export const getByMonthDetail = (
  detail: PortfolioBalanceEstimation.Detail,
  ianaTimezoneName: string,
) => {
  const result = (x: typeof monthsDesc) => ({ ...detail, monthsDesc: x })

  if (detail.actions.length === 0) return result([])
  const getZonedTime = getZonedTimeFns(ianaTimezoneName)
  const actionsDesc = _.reverse(detail.actions.slice())

  const actionsDescByMonthMap = groupBy(actionsDesc, (x) =>
    getZonedTime(x.timestamp).startOf('month').toMillis(),
  )

  let currState = fGet(_.first(actionsDesc)).stateChange.end

  const monthsDesc = _getDescTimeSeq(
    ianaTimezoneName,
    detail.endTimestamp,
    detail.startTimestamp,
    'month',
  )
    .map((month) => {
      const actionsDesc = actionsDescByMonthMap.get(month) ?? []
      actionsDescByMonthMap.delete(month)
      return { month, actionsDesc }
    })
    .map(({ month, actionsDesc }, i) => {
      const actionsDescByDayMap = groupBy(actionsDesc, (action) =>
        getZonedTime(action.timestamp).startOf('day').toMillis(),
      )

      const daysDesc = _getDescTimeSeq(
        ianaTimezoneName,
        Math.min(
          getZonedTime(month).endOf('month').toMillis(),
          detail.endTimestamp,
        ),
        Math.max(month, detail.startTimestamp),
        'day',
      ).map((day) => {
        const actionsDesc = actionsDescByDayMap.get(day) ?? []
        actionsDescByDayMap.delete(day)

        return { day, actionsDesc }
      })
      assert(actionsDescByDayMap.size === 0)

      const stateChange = {
        end: currState,
        start: _.last(actionsDesc)?.stateChange.start ?? currState,
      }
      currState = stateChange.start
      return { month, stateChange, daysDesc }
    })
  assert(actionsDescByMonthMap.size === 0)
  return result(monthsDesc)
}



const _getDescTimeSeq = (
  ianaTimezoneName: string,
  end: number,
  startIn: number,
  delta: 'day' | 'month',
) => {
  const getZonedTime = getZonedTimeFns(ianaTimezoneName)
  const result = [] as number[]
  const start = getZonedTime(startIn).startOf(delta)
  let curr = getZonedTime(end).startOf(delta)
  while (curr >= start) {
    result.push(curr.toMillis())
    curr = curr.minus({ [delta]: 1 })
  }
  return result
}