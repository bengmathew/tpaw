import { faLeftLong } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { ReactNode } from 'react'
import { FirstMonthSavingsPortfolioDetail } from '../../TPAWSimulator/Worker/FirstMonthSavingsPortfolioDetail'
import { TPAWRunInWorkerByPercentileByMonthsFromNow } from '../../TPAWSimulator/Worker/TPAWRunInWorker'
import { UseTPAWWorkerResult } from '../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { getPrecision } from '../../Utils/GetPrecision'
import { useURLUpdater } from '../../Utils/UseURLUpdater'
import { fGet } from '../../Utils/Utils'
import { AppPage } from '../App/AppPage'
import { useSimulation } from '../App/WithSimulation'
import { useGetSectionURL } from '../Plan/Plan'
import { PlanSectionName } from '../Plan/PlanInput/Helpers/PlanSectionName'

export type TasksForThisMonthProps = Omit<
  FirstMonthSavingsPortfolioDetail,
  'withdrawals'
> & {
  withdrawals: FirstMonthSavingsPortfolioDetail['withdrawals'] & {
    essentialByEntry: { id: number; label: string | null; amount: number }[]
    discretionaryByEntry: { id: number; label: string | null; amount: number }[]
  }
  withdrawalsStarted: boolean
  strategy: PlanParams['advanced']['strategy']
}

export const getTasksForThisMonthProps = (
  tpawResult: UseTPAWWorkerResult,
): TasksForThisMonthProps => {
  const original = tpawResult.firstMonthOfSomeRun
  const { params, paramsExt } = tpawResult

  const { withdrawalsStarted } = paramsExt

  const firstMonthOfAnyPercentile = (
    id: number,
    { byId }: { byId: Map<number, TPAWRunInWorkerByPercentileByMonthsFromNow> },
  ) => fGet(byId.get(id)).byPercentileByMonthsFromNow[0].data[0]

  return {
    ...original,
    withdrawals: {
      ...original.withdrawals,
      essentialByEntry:
        params.original.plan.adjustmentsToSpending.extraSpending.essential.map(
          ({ id, label }) => ({
            id,
            label,
            amount: firstMonthOfAnyPercentile(
              id,
              tpawResult.savingsPortfolio.withdrawals.essential,
            ),
          }),
        ),
      discretionaryByEntry:
        params.original.plan.adjustmentsToSpending.extraSpending.discretionary.map(
          ({ id, label }) => ({
            id,
            label,
            amount: firstMonthOfAnyPercentile(
              id,
              tpawResult.savingsPortfolio.withdrawals.discretionary,
            ),
          }),
        ),
    },
    withdrawalsStarted,
    strategy: params.strategy,
  }
}

export const TasksForThisMonth = React.memo(() => {
  const getSectionURL = useGetSectionURL()
  const urlUpdater = useURLUpdater()

  return (
    <AppPage
      className="pt-header min-h-screen bg-gray-300"
      title="Tasks for This Month - TPAW Planner"
      curr="plan"
    >
      <div className="flex flex-col items-center  mt-6 px-4 ">
        <div className="w-full max-w-[650px] px-4 py-4 z-0 bg-cardBG rounded-2xl mb-20 ">
          <div className="">
            <div className=" flex flex-row items-center gap-x-4 gap-y-">
              <button
                className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5"
                onClick={() =>
                  urlUpdater.push(
                    getSectionURL(getTasksForThisMonthOnDoneSection()),
                  )
                }
              >
                <FontAwesomeIcon className="" icon={faLeftLong} />
                Done
              </button>
              <h1 className="font-bold text-3xl">Tasks for This Month</h1>
            </div>
            <TasksForThisMonthContent className="mt-10" />
          </div>
        </div>
      </div>
    </AppPage>
  )
})

export const TasksForThisMonthContent = React.memo(
  ({
    className,
    forPrint = false,
  }: {
    className?: string
    forPrint?: boolean
  }) => {
    const { tpawResult } = useSimulation()
    const props = getTasksForThisMonthProps(tpawResult)

    const { withdrawals, withdrawalsStarted } = props
    return (
      <div className={clsx(className, forPrint ? '' : 'p-base')}>
        {withdrawals.total !== 0 || withdrawalsStarted ? (
          <>
            <h2
              className={clsx(
                'font-bold text-2xl mb-4 font-font1 ',
                forPrint ? 'text-black' : 'text-pageFG',
              )}
            >
              Summary
            </h2>
            <_Summary className="" {...props} forPrint={forPrint} />

            <h2
              className={clsx(
                'font-bold text-2xl mt-10 mb-6  font-font1 ',
                forPrint ? 'text-black' : 'text-pageFG',
              )}
            >
              Details
            </h2>
            <_Details className="" {...props} forPrint={forPrint} />
          </>
        ) : (
          <_Details className="" {...props} forPrint={forPrint} />
        )}
      </div>
    )
  },
)

// -------- SUMMARY --------
const _Summary = React.memo(
  ({
    className,
    forPrint,
    ...props
  }: TasksForThisMonthProps & { className: string; forPrint: boolean }) => {
    const { contributionToOrWithdrawalFromSavingsPortfolio } = props
    const withdrawalText =
      contributionToOrWithdrawalFromSavingsPortfolio.type === 'withdrawal'
        ? `Withdraw ${formatCurrency(
            contributionToOrWithdrawalFromSavingsPortfolio.withdrawal,
          )} from `
        : `Contribute ${formatCurrency(
            Math.abs(
              contributionToOrWithdrawalFromSavingsPortfolio.contribution,
            ),
          )} to`

    return (
      <div className={className}>
        <div className="mt-2">
          <p className="mb-2">
            {withdrawalText} your portfolio and rebalance your remaining
            portfolio of <_Value>{props.afterWithdrawals.balance}</_Value> to
            the following asset allocation:
          </p>
          <_AllocationTable className="" {...props} />
        </div>
      </div>
    )
  },
)

//  -------- DETAILS --------
const _Details = React.memo(
  ({
    className,
    forPrint,
    ...props
  }: TasksForThisMonthProps & { className: string; forPrint: boolean }) => {
    const {
      withdrawals,
      contributionToOrWithdrawalFromSavingsPortfolio,
      withdrawalsStarted,
    } = props
    return (
      <div className={className}>
        {(withdrawals.total !== 0 || withdrawalsStarted) && (
          <>
            <_HowMuchYouHave
              className="mb-8 break-inside-avoid-page"
              forPrint={forPrint}
              {...props}
            />
            <_HowMuchToSpend
              className="mb-8  break-inside-avoid-page"
              forPrint={forPrint}
              {...props}
            />
            {withdrawals.total > 0 && (
              <_HowToFundTheSpending
                forPrint={forPrint}
                className="mb-8  break-inside-avoid-page"
                {...props}
              />
            )}
          </>
        )}
        {(!withdrawalsStarted ||
          contributionToOrWithdrawalFromSavingsPortfolio.type ===
            'contribution') && (
          <_Contribution className="mb-8" forPrint={forPrint} {...props} />
        )}
        <_AssetAllocation
          className=" break-inside-avoid-page"
          forPrint={forPrint}
          {...props}
        />
      </div>
    )
  },
)

// -------- HOW MUCH YOU HAVE --------
const _HowMuchYouHave = React.memo(
  ({
    withdrawalsStarted,
    contributions,
    className = '',
    forPrint,
    ...props
  }: TasksForThisMonthProps & { className?: string; forPrint: boolean }) => {
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>How Much You Have</_Heading>
        <p className="mb-4">
          Your current portfolio is <_Value>{props.start.balance}</_Value> and
          you{' '}
          {!withdrawalsStarted ? (
            <span>
              plan to contribute and additional{' '}
              <_Value>{contributions.total}</_Value>
            </span>
          ) : (
            <span>
              have <_Value>{contributions.total}</_Value> in retirement income
            </span>
          )}{' '}
          this month. This gives you a total of{' '}
          <_Value className="">{props.afterContributions.balance}</_Value> to
          split between spending for this month and saving for the future.
        </p>
      </div>
    )
  },
)

// -------- HOW MUCH TO SPEND --------
const _HowMuchToSpend = React.memo(
  ({
    withdrawals,
    className = '',
    forPrint,
    ...props
  }: TasksForThisMonthProps & { className?: string; forPrint: boolean }) => {
    const needsBreakdown =
      withdrawals.essentialByEntry.length > 0 ||
      withdrawals.discretionaryByEntry.length > 0
    const mergeBreakdown = props.strategy === 'SWR'
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>How Much To Spend</_Heading>
        <p className="">
          Out of your <_Value>{props.start.balance}</_Value>, you plan to spend{' '}
          <_Value className="">{withdrawals.total}</_Value> this month
          {needsBreakdown
            ? `,
          broken down as follows:`
            : '.'}
        </p>
        {needsBreakdown && (
          <div
            className="grid justify-start gap-x-10 mt-2 "
            style={{ grid: 'auto/auto auto' }}
          >
            <h2 className="">Regular</h2>
            <h2 className="text-right">
              <_Value>{withdrawals.regular}</_Value>
            </h2>
            {mergeBreakdown ? (
              <>
                <h2 className="">Extra</h2>
                <h2></h2>
                {withdrawals.essentialByEntry.map(({ id, label, amount }) => (
                  <React.Fragment key={id}>
                    <h2 className="ml-8">{label ?? '<no label>'}</h2>
                    <h2 className="text-right">
                      <_Value>{amount}</_Value>
                    </h2>
                  </React.Fragment>
                ))}
                {withdrawals.discretionaryByEntry.map(
                  ({ id, label, amount }) => (
                    <React.Fragment key={id}>
                      <h2 className="ml-8">{label ?? '<no label>'}</h2>
                      <h2 className="text-right">
                        <_Value>{amount}</_Value>
                      </h2>
                    </React.Fragment>
                  ),
                )}
              </>
            ) : (
              <>
                {withdrawals.essentialByEntry.length > 0 && (
                  <>
                    <h2 className="">Extra - Essential</h2>
                    <h2></h2>
                    {withdrawals.essentialByEntry.map(
                      ({ id, label, amount }) => (
                        <React.Fragment key={id}>
                          <h2 className="ml-8">{label ?? '<no label>'}</h2>
                          <h2 className="text-right">
                            <_Value>{amount}</_Value>
                          </h2>
                        </React.Fragment>
                      ),
                    )}
                  </>
                )}
                {withdrawals.discretionaryByEntry.length > 0 && (
                  <>
                    <h2 className="">Extra - Discretionary</h2>
                    <h2></h2>
                    {withdrawals.discretionaryByEntry.map(
                      ({ id, label, amount }) => (
                        <React.Fragment key={id}>
                          <h2 className="ml-8">{label ?? '<no label>'}</h2>
                          <h2 className="text-right">
                            <_Value>{amount}</_Value>
                          </h2>
                        </React.Fragment>
                      ),
                    )}
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
    )
  },
)

// -------- HOW TO FUND THE SPENDING --------
const _HowToFundTheSpending = React.memo(
  ({
    withdrawals,
    withdrawalsStarted,
    forPrint,
    className = '',
  }: TasksForThisMonthProps & { className?: string; forPrint: boolean }) => {
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>How To Fund The Spending</_Heading>
        <p className="mb-3">
          Fund your spending of <_Value>{withdrawals.total}</_Value> from the
          following sources:{' '}
        </p>
        <div
          className="grid  justify-start gap-x-6 "
          style={{ grid: 'auto/auto auto' }}
        >
          <h2 className="">Withdrawal from Portfolio</h2>
          <h2 className="text-right">
            <_Value>{withdrawals.fromSavingsPortfolio}</_Value>
          </h2>
          <h2 className="">
            {withdrawalsStarted ? 'Retirement Income' : 'Planned Savings'}
          </h2>
          <h2 className="text-right">
            <_Value>{withdrawals.fromContributions}</_Value>
          </h2>
        </div>
      </div>
    )
  },
)

// -------- CONTRIBUTION --------
const _Contribution = React.memo(
  ({
    contributions,
    contributionToOrWithdrawalFromSavingsPortfolio,
    withdrawalsStarted,
    forPrint,
    className = '',
  }: TasksForThisMonthProps & { className?: string; forPrint: boolean }) => {
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>
          How Much To Contribute To Your Portfolio
        </_Heading>
        {contributionToOrWithdrawalFromSavingsPortfolio.type ===
        'contribution' ? (
          contributions.toWithdrawal > 0 ? (
            <p className="mb-3 ">
              Your{' '}
              {withdrawalsStarted ? 'retirement income' : 'planned savings'} for
              this month is <_Value>{contributions.total}</_Value> You used{' '}
              <_Value>{contributions.toWithdrawal}</_Value> of this to fund
              spending for this month. Contribute the remaining{' '}
              <_Value className="">{contributions.toSavingsPortfolio}</_Value>{' '}
              to your portfolio.{' '}
            </p>
          ) : (
            <p className="mb-3 ">
              Your{' '}
              {withdrawalsStarted ? 'retirement income' : 'planned savings'} for
              this month is <_Value>{contributions.total}</_Value>. Contribute
              all <_Value className="">{contributions.total}</_Value> to your
              portfolio.{' '}
            </p>
          )
        ) : (
          <p className="mb-3 ">
            You do not have a contribution to your portfolio this month.
          </p>
        )}
      </div>
    )
  },
)

// -------- ASSET ALLOCATION --------
const _AssetAllocation = React.memo(
  ({
    className = '',
    forPrint,
    ...props
  }: TasksForThisMonthProps & { className?: string; forPrint: boolean }) => {
    const { contributionToOrWithdrawalFromSavingsPortfolio } = props
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>What Asset Allocation To Use</_Heading>
        <div className="mb-3 ">
          <p className="mb-3">
            You started with a portfolio of{' '}
            <_Value>{props.start.balance}</_Value> and{' '}
            {contributionToOrWithdrawalFromSavingsPortfolio.type ===
            'withdrawal' ? (
              <span>
                withdrew{' '}
                <_Value>
                  {contributionToOrWithdrawalFromSavingsPortfolio.withdrawal}
                </_Value>{' '}
                from it.
              </span>
            ) : (
              <span>
                contributed{' '}
                <_Value>
                  {contributionToOrWithdrawalFromSavingsPortfolio.contribution}
                </_Value>{' '}
                to it.
              </span>
            )}{' '}
            The portfolio after the{' '}
            {contributionToOrWithdrawalFromSavingsPortfolio.type ===
            'withdrawal'
              ? 'withdrawal'
              : 'contribution'}{' '}
            is <_Value className="">{props.afterWithdrawals.balance}</_Value>.
            Rebalance this portfolio to the following asset allocation:
          </p>
          <_AllocationTable className="" {...props} />
        </div>
      </div>
    )
  },
)

const _AllocationTable = React.memo(
  ({
    className = '',
    ...props
  }: TasksForThisMonthProps & { className?: string }) => {
    const { allocation } = props.afterWithdrawals
    const stocks = props.afterWithdrawals.balance * allocation.stocks
    const bonds = props.afterWithdrawals.balance - stocks
    // Using full precision in display because rounding should happen at the
    // data level.
    console.dir(allocation.stocks)
    const stocksFullStr = formatPercentage('full')(allocation.stocks)
    const bondsFullStr = formatPercentage('full')(
      _.round(1 - allocation.stocks, getPrecision(allocation.stocks)),
    )
    return (
      <div
        className={`${className} inline-grid justify-start gap-x-2`}
        style={{ grid: 'auto/auto auto auto auto' }}
      >
        <h2 className="">Stocks</h2>
        <h2 className="ml-8 text-right">
          <_Value>{stocks}</_Value>
        </h2>
        <h2 className="text-right ml-4">
          {formatPercentage(0)(allocation.stocks)}
        </h2>
        {stocksFullStr.indexOf('.') !== -1 ? (
          <h2 className="text-right">({stocksFullStr})</h2>
        ) : (
          <h2></h2>
        )}
        <h2 className="">Bonds</h2>
        <h2 className="ml-8 text-right">
          <_Value>{bonds}</_Value>
        </h2>
        <h2 className="text-right ml-4">
          {formatPercentage(0)(1 - allocation.stocks)}
        </h2>
        {bondsFullStr.indexOf('.') !== -1 ? (
          <h2 className="text-right">({bondsFullStr})</h2>
        ) : (
          <h2></h2>
        )}
      </div>
    )
  },
)

const _Heading = React.memo(
  ({ children, forPrint }: { children: ReactNode; forPrint: boolean }) => (
    <h2
      className={clsx(
        'font-bold text-lg mb-1 font-font1 ',
        forPrint ? 'text-black' : 'text-pageFG',
      )}
    >
      {children}
    </h2>
  ),
)

const _Value = React.memo(
  ({ children, className }: { children: number; className?: string }) => (
    <span className={className}>{formatCurrency(children)}</span>
  ),
)

export const setTasksForThisMonthOnDoneSection = (section: PlanSectionName) => {
  window.localStorage.setItem('TasksForThisMonthOnDoneSection', section)
}
const getTasksForThisMonthOnDoneSection = (): PlanSectionName => {
  const result = window.localStorage.getItem(
    'TasksForThisMonthOnDoneSection',
  ) as PlanSectionName | undefined

  window.localStorage.removeItem('TasksForThisMonthOnDoneSection')
  return result ?? ('summary' as PlanSectionName)
}
