import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faTurnDownLeft } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { block, getZonedTimeFns, noCase } from '@tpaw/common'
import { default as clix, default as clsx } from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../Utils/Geometry'
import { Spinner } from '../../../../Utils/View/Spinner'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { CurrentPortfolioBalance } from '../../PlanRootHelpers/CurrentPortfolioBalance'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { planSectionLabel } from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

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
    const [showExplanation, setShowHistory] = useState(false)
    const { currentPortfolioBalanceInfo, updatePlanParams } = useSimulation()
    const amountInfo = CurrentPortfolioBalance.getAmountInfo(
      currentPortfolioBalanceInfo,
    )

    const handleChange = (amount: number) =>
      updatePlanParams('setCurrentPortfolioBalance', amount)

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
            <AmountInput
              className="text-input"
              prefix="$"
              value={amountInfo.amount}
              onChange={(x) => {
                // This happens on blur and enter. The check for real change is
                // especially important when amountInfo.amount is an estimate when
                // handleChange() is called, we mark it as not a estimate, which
                // is materially different. We don't want to do that on blur and
                // enter if the value has not changed.
                if (x !== amountInfo.amount) handleChange(x)
              }}
              decimals={0}
              modalLabel={planSectionLabel('current-portfolio-balance')}
            />
            <button
              className="ml-2 px-3"
              onClick={() => {
                handleChange(
                  smartDeltaFnForAmountInput.increment(amountInfo.amount),
                )
              }}
            >
              <FontAwesomeIcon icon={faPlus} />
            </button>
            <button
              className="px-3"
              onClick={() => {
                handleChange(
                  smartDeltaFnForAmountInput.decrement(amountInfo.amount),
                )
              }}
            >
              <FontAwesomeIcon icon={faMinus} />
            </button>
          </div>
          {amountInfo.isEstimate &&
            amountInfo.lastEnteredAmount !== amountInfo.amount && (
              <div className={'mt-4 ml-2 flex items-top gap-x-1'}>
                <FontAwesomeIcon
                  className="rotate-90 mr-1"
                  icon={faTurnDownLeft}
                />
                <div className="">
                  <p className="p-base">
                    This is an estimate calculated from your last entry of{' '}
                    {formatCurrency(amountInfo.lastEnteredAmount)} on{' '}
                    {getZonedTimeFns(amountInfo.ianaTimezoneName)(
                      amountInfo.lastEnteredTimestamp,
                    ).toLocaleString(DateTime.DATE_MED)}
                    .{' '}
                  </p>
                  <button
                    className="underline block mt-1"
                    onClick={() => setShowHistory(true)}
                  >
                    View Balance History
                  </button>
                </div>
              </div>
            )}
        </div>
        <CenteredModal
          className=" dialog-outer-div"
          show={showExplanation}
          onOutsideClickOrEscape={() => setShowHistory(false)}
        >
          <_Popup />
        </CenteredModal>
      </div>
    )
  },
)

const _Popup = React.memo(() => {
  const { currentTimestamp, simulationInfoBySrc, currentPortfolioBalanceInfo } =
    useSimulation()
  const amountInfo = CurrentPortfolioBalance.getAmountInfo(
    currentPortfolioBalanceInfo,
  )
  const historyModeInfo = block(() => {
    switch (simulationInfoBySrc.src) {
      case 'link':
      case 'localMain':
        return 'fetched' as const
      default:
        return simulationInfoBySrc.historyStatus
    }
  })
  const byMonth = useMemo(
    () =>
      CurrentPortfolioBalance.mergeByMonthInfo(
        ..._.compact([
          currentPortfolioBalanceInfo.preBase,
          currentPortfolioBalanceInfo.postBase
            ? CurrentPortfolioBalance.getByMonthInfo(
                currentPortfolioBalanceInfo.postBase,
              )
            : null,
        ]),
      ),
    [currentPortfolioBalanceInfo.postBase, currentPortfolioBalanceInfo.preBase],
  )
  const { getZonedTime } = useIANATimezoneName()

  const formatTime = (x: number) => getZonedTime(x).toFormat('LLLL d, yyyy')
  return (
    <div className="p-2 min-w-[600px] sm:min-w-auto">
      {/* <h2 className="text-right">Zone: {currentTime.toFormat('ZZZZ')}</h2> */}
      <div className="ml-4">
        {/* <h2 className="font-bold text-lg">You Entered</h2>
          <div className="">
            <h2>
              {' '}
              {formatCurrency(start.amount)} on {formatTime(start.timestamp)}
            </h2>
          </div> */}
        <h2 className="font-bold mt-2 text-lg">Current Estimate</h2>
        <div className="">
          <h2>
            {' '}
            <span className="">{formatCurrency(amountInfo.amount)}</span> as of{' '}
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

      {byMonth.monthsDesc.map((info) => (
        <_Month
          key={`month-${info.month}`}
          className="mt-4 mb-10 rounded-lg  bg-gray-100 px-4 pb-4"
          info={info}
        />
      ))}
      {historyModeInfo === 'fetched' ? (
        <></>
      ) : historyModeInfo === 'failed' ? (
        <div className="mt-5 text-errorFG text-center">
          Failed to fetch full history. Please reload to try again.
        </div>
      ) : historyModeInfo === 'fetching' ? (
        <div className="relative h-[50px]">
          <Spinner size="text-3xl" />
        </div>
      ) : (
        noCase(historyModeInfo)
      )}
    </div>
  )
})

type MonthInfo = CurrentPortfolioBalance.ByMonthInfo['monthsDesc'][0]
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
    action: CurrentPortfolioBalance.Action
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
            case 'withdarwalAndContribution':
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
  ({
    amountInfo,
    forPrint,
  }: {
    amountInfo: CurrentPortfolioBalance.AmountInfo
    forPrint: boolean
  }) => {
    return (
      <>
        <h2>{formatCurrency(amountInfo.amount)}</h2>
        {amountInfo.isEstimate &&
          amountInfo.lastEnteredAmount !== amountInfo.amount && (
            <div className={'ml-2 flex items-top gap-x-1'}>
              <FontAwesomeIcon
                className={clsx("rotate-90 mr-1", forPrint && 'text-[10px]')}
                icon={faTurnDownLeft}
              />
              <div className="">
                <p className={clsx(forPrint ? '' : 'font-font2 text-base')}>
                  This is an estimate calculated from your last entry of{' '}
                  {formatCurrency(amountInfo.lastEnteredAmount)} on{' '}
                  {getZonedTimeFns(amountInfo.ianaTimezoneName)(
                    amountInfo.lastEnteredTimestamp,
                  ).toLocaleString(DateTime.DATE_MED)}
                  .{' '}
                </p>
              </div>
            </div>
          )}
      </>
    )
  },
)
