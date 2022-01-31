import React, { ReactNode } from 'react'
import { TPAWSimulationForYear } from '../../TPAWSimulator/RunTPAWSimulation'
import { UseTPAWWorkerResult } from '../../TPAWSimulator/Worker/UseTPAWWorker'
import { formatCurrency } from '../../Utils/FormatCurrency'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { AppPage } from '../App/AppPage'
import { Footer } from '../App/Footer'
import { Header } from '../App/Header'
// import { useTPAW } from '../Common/UseTPAW/UseTPAW'
import { useSimulation } from '../App/WithSimulation'

type _Props = {
  withdrawal: TPAWSimulationForYear['withdrawalAchieved'] & {
    src: {fromSavings: number; fromIncome: number}
  }
  allocation: TPAWSimulationForYear['savingsPortfolioAllocation']
  isWithdrawal: boolean
  isRetired: boolean
}
const _getProps = (tpawResult: UseTPAWWorkerResult): _Props => {
  const {age} = tpawResult.args.params
  const {
    withdrawalAchieved: withdrawal,
    savingsPortfolioAllocation: allocation,
  } = tpawResult.firstYearOfSomeRun

  return {
    withdrawal: {
      ...withdrawal,
      src: (() => {
        const fromSavings = Math.max(0, withdrawal.fromSavings)
        return {fromSavings, fromIncome: withdrawal.total - fromSavings}
      })(),
    },
    allocation: allocation,
    isWithdrawal: withdrawal.fromSavings > 0,
    isRetired: age.retirement === age.start,
  }
}

export const TasksForThisYear = React.memo(() => {
  const {tpawResult} = useSimulation()

  const props = _getProps(tpawResult)

  const {withdrawal, isRetired} = props

  return (
    <AppPage
      className="grid pt-header"
      title="Tasks for This Year - TPAW Planner"
      style={{grid: '1fr auto/auto'}}
      curr="plan"
    >
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-2 z-0">
          <div className="">
            <h1 className="font-bold text-4xl ">Tasks for This Year</h1>

            {withdrawal.total !== 0 || isRetired ? (
              <>
                <h2 className="font-bold text-2xl mt-10 mb-4">Summary</h2>
                <_Summary className="" {...props} />

                <h2 className="font-bold text-2xl mt-10 mb-6">Details</h2>
                <_Details className="" {...props} />
              </>
            ) : (
              <_Details className="mt-10" {...props} />
            )}
          </div>
        </div>
      </div>
      <Footer />
    </AppPage>
  )
})

// -------- SUMMARY --------
const _Summary = React.memo(
  ({className, ...props}: _Props & {className: string}) => {
    const {isWithdrawal, withdrawal, allocation} = props
    const withdrawalText = isWithdrawal
      ? `Withdraw ${formatCurrency(withdrawal.fromSavings)} from `
      : `Contribute ${formatCurrency(Math.abs(withdrawal.fromSavings))} to`

    return (
      <div className={className}>
        <div className="mt-2 p-base">
          <p className="mb-2">
            {withdrawalText} your portfolio and rebalance your portfolio of{' '}
            <_Value>{allocation.balance}</_Value> to the following asset
            allocation:
          </p>
          <_AllocationTable className="" {...props} />
        </div>
      </div>
    )
  }
)

//  -------- DETAILS --------
const _Details = React.memo(
  ({className, ...props}: _Props & {className: string}) => {
    const {withdrawal, isRetired} = props
    return (
      <div className={className}>
        {(withdrawal.total !== 0 || isRetired) && (
          <>
            <_HowMuchYouHave className="mb-8" {...props} />
            <_HowMuchToSpend className="mb-8" {...props} />
            {withdrawal.total > 0 && (
              <_HowToFundTheSpending className="mb-8" {...props} />
            )}
          </>
        )}
        {(!isRetired || withdrawal.fromSavings < 0) && (
          <_Contribution className="mb-8" {...props} />
        )}
        <_AssetAllocation className="mb-8" {...props} />
      </div>
    )
  }
)

// -------- HOW MUCH YOU HAVE --------
const _HowMuchYouHave = React.memo(
  ({withdrawal, isRetired, className = ''}: _Props & {className?: string}) => (
    <div className={className}>
      <_Heading>How Much You Have</_Heading>
      <p className="mb-4 p-base">
        Your current portfolio is <_Value>{withdrawal.savings}</_Value> and you{' '}
        {!isRetired ? (
          <span>
            plan to contribute and additional{' '}
            <_Value>{withdrawal.currentYearIncome}</_Value>
          </span>
        ) : (
          <span>
            have <_Value>{withdrawal.currentYearIncome}</_Value> in retirement
            income
          </span>
        )}{' '}
        this year. This gives you a total of{' '}
        <_Value className="">{withdrawal.availableFunds}</_Value> to split
        between spending for this year and saving for the future.
      </p>
    </div>
  )
)

// -------- HOW MUCH TO SPEND --------
const _HowMuchToSpend = React.memo(
  ({withdrawal, className = ''}: _Props & {className?: string}) => (
    <div className={className}>
      <_Heading>How Much To Spend</_Heading>
      <p className="p-base">
        Out of your <_Value>{withdrawal.availableFunds}</_Value>, you plan to
        spend <_Value className="">{withdrawal.total}</_Value> this year, broken
        down as follows:
      </p>
      <div
        className="grid justify-start gap-x-10 mt-2 p-base"
        style={{grid: 'auto/auto auto'}}
      >
        <h2 className="">Extra - Essential</h2>
        <h2 className="text-right">
          <_Value>{withdrawal.essentail}</_Value>
        </h2>
        <h2 className="">Extra - Discretionary</h2>
        <h2 className="text-right">
          <_Value>{withdrawal.extra}</_Value>
        </h2>
        <h2 className="">Regular</h2>
        <h2 className="text-right">
          <_Value>{withdrawal.regular}</_Value>
        </h2>
      </div>
    </div>
  )
)

// -------- HOW TO FUND THE SPENDING --------
const _HowToFundTheSpending = React.memo(
  ({withdrawal, isRetired, className = ''}: _Props & {className?: string}) => (
    <div className={className}>
      <_Heading>How To Fund The Spending</_Heading>
      <p className="mb-3 p-base">
        Fund your spending of <_Value>{withdrawal.total}</_Value> from the
        following sources:{' '}
      </p>
      <div
        className="grid  justify-start gap-x-6 p-base"
        style={{grid: 'auto/auto auto'}}
      >
        <h2 className="">Withdrawal from Portfolio</h2>
        <h2 className="text-right">
          <_Value>{withdrawal.src.fromSavings}</_Value>
        </h2>
        <h2 className="">
          {isRetired ? 'Retirement Income' : 'Planned Savings'}
        </h2>
        <h2 className="text-right">
          <_Value>{withdrawal.src.fromIncome}</_Value>
        </h2>
      </div>
    </div>
  )
)

// -------- CONTRIBUTION --------
const _Contribution = React.memo(
  ({withdrawal, isRetired, className = ''}: _Props & {className?: string}) => (
    <div className={className}>
      <_Heading>How Much To Contribute To Your Portfolio</_Heading>
      {withdrawal.fromSavings < 0 ? (
        withdrawal.src.fromIncome > 0 ? (
          <p className="mb-3 p-base">
            Your {isRetired ? 'retirement income' : 'planned savings'} for this
            year is <_Value>{withdrawal.currentYearIncome}</_Value> You used{' '}
            <_Value>{withdrawal.src.fromIncome}</_Value> of this to fund
            spending for this year. Contribute the remaining{' '}
            <_Value className="">{Math.abs(withdrawal.fromSavings)}</_Value> to
            your portfolio.{' '}
          </p>
        ) : (
          <p className="mb-3 p-base">
            Your {isRetired ? 'retirement income' : 'planned savings'} for this
            year is <_Value>{withdrawal.currentYearIncome}</_Value>. Contribute
            all <_Value className="">{Math.abs(withdrawal.fromSavings)}</_Value>{' '}
            to your portfolio.{' '}
          </p>
        )
      ) : (
        <p className="mb-3 p-base">
          You do not have a contribution to your portfolio this year.
        </p>
      )}
    </div>
  )
)

// -------- ASSET ALLOCATION --------
const _AssetAllocation = React.memo(
  ({className = '', ...props}: _Props & {className?: string}) => {
    const {withdrawal, isWithdrawal, allocation} = props
    return (
      <div className={className}>
        <_Heading>What Asset Allocation To Use</_Heading>
        <div className="mb-3 p-base">
          <p className="mb-3">
            You started with a portfolio of{' '}
            <_Value>{withdrawal.savings}</_Value> and{' '}
            {isWithdrawal ? (
              <span>
                withdrew <_Value>{withdrawal.fromSavings}</_Value> from it.
              </span>
            ) : (
              <span>
                contributed <_Value>{Math.abs(withdrawal.fromSavings)}</_Value>{' '}
                to it.
              </span>
            )}{' '}
            The portfolio after the{' '}
            {isWithdrawal ? 'withdrawal' : 'contribution'} is{' '}
            <_Value className="">{allocation.balance}</_Value>. Rebalance the
            portfolio to the following asset allocation:
          </p>
          <_AllocationTable className="" {...props} />
        </div>
      </div>
    )
  }
)

const _AllocationTable = React.memo(
  ({className = '', allocation}: _Props & {className?: string}) => (
    <div
      className={`${className} grid justify-start gap-x-16`}
      style={{grid: 'auto/auto auto'}}
    >
      <h2 className="">
        Stocks - {formatPercentage(0)(allocation.asPercentage.stocks ?? 0)}
      </h2>
      <h2 className="text-right">
        <_Value>{allocation.stocksAchieved}</_Value>
      </h2>
      <h2 className="">
        Bonds - {formatPercentage(0)(allocation.asPercentage.bonds ?? 0)}
      </h2>
      <h2 className="text-right">
        <_Value>{allocation.bonds}</_Value>
      </h2>
    </div>
  )
)

const _Heading = React.memo(({children}: {children: ReactNode}) => (
  <h2 className="font-bold text-lg mb-1">{children}</h2>
))

const _Value = React.memo(
  ({children, className}: {children: number; className?: string}) => (
    <span className={className}>{formatCurrency(children)}</span>
  )
)

export const mainPagePlanForThisYearSummaryText = (
  tpawResult: UseTPAWWorkerResult
) => {
  const {isWithdrawal, withdrawal, allocation} = _getProps(tpawResult)
  const withdrawalSubHeading = isWithdrawal
    ? `Withdraw ${formatCurrency(withdrawal.fromSavings)}`
    : `Contribute ${formatCurrency(Math.abs(withdrawal.fromSavings))}`

  return `${withdrawalSubHeading}. Set portfolio to ${formatPercentage(0)(
    allocation.asPercentage.stocks ?? 0
  )} stocks.`
}
