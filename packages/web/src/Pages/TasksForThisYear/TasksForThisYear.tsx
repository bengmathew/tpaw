import {faLeftLong} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, {ReactNode} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../TPAWSimulator/TPAWParamsExt'
import {FirstYearSavingsPortfolioDetail} from '../../TPAWSimulator/Worker/FirstYearSavingsPortfolioDetail'
import {TPAWRunInWorkerByPercentileByYearsFromNow} from '../../TPAWSimulator/Worker/TPAWRunInWorker'
import {UseTPAWWorkerResult} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {formatCurrency} from '../../Utils/FormatCurrency'
import {formatPercentage} from '../../Utils/FormatPercentage'
import {fGet} from '../../Utils/Utils'
import {AppPage} from '../App/AppPage'
import {Footer} from '../App/Footer'
import {useSimulation} from '../App/WithSimulation'
import {paramsInputLabel} from '../Plan/ParamsInput/Helpers/ParamsInputLabel'

type _Props = Omit<FirstYearSavingsPortfolioDetail, 'withdrawals'> & {
  withdrawals: FirstYearSavingsPortfolioDetail['withdrawals'] & {
    essentialByEntry: {id: number; label: string | null; amount: number}[]
    discretionaryByEntry: {id: number; label: string | null; amount: number}[]
  }
  withdrawalsStarted: boolean
  strategy: TPAWParams['strategy']
}

const _getProps = (tpawResult: UseTPAWWorkerResult): _Props => {
  const original = tpawResult.firstYearOfSomeRun
  const {params} = tpawResult.args

  const {withdrawalsStarted} = extendTPAWParams(params.original)

  const firstYearOfAnyPercentile = (
    id: number,
    {byId}: {byId: Map<number, TPAWRunInWorkerByPercentileByYearsFromNow>}
  ) => fGet(byId.get(id)).byPercentileByYearsFromNow[0].data[0]

  return {
    ...original,
    withdrawals: {
      ...original.withdrawals,
      essentialByEntry: params.withdrawals.essential.map(({id, label}) => ({
        id,
        label,
        amount: firstYearOfAnyPercentile(
          id,
          tpawResult.savingsPortfolio.withdrawals.essential
        ),
      })),
      discretionaryByEntry: params.withdrawals.discretionary.map(
        ({id, label}) => ({
          id,
          label,
          amount: firstYearOfAnyPercentile(
            id,
            tpawResult.savingsPortfolio.withdrawals.discretionary
          ),
        })
      ),
    },
    withdrawalsStarted,
    strategy: params.strategy,
  }
}

export const TasksForThisYear = React.memo(() => {
  const {tpawResult} = useSimulation()
  const props = _getProps(tpawResult)
  const {withdrawals, withdrawalsStarted} = props

  return (
    <AppPage
      className="grid pt-header min-h-screen"
      title="Tasks for This Year - TPAW Planner"
      style={{grid: '1fr auto/auto'}}
      curr="plan"
    >
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-2 z-0">
          <div className="">
            <div className=" flex flex-row items-center gap-x-4 gap-y-">
              <Link href="/plan">
                <a className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5">
                  <FontAwesomeIcon className="" icon={faLeftLong} />
                  Done
                </a>
              </Link>
              <h1 className="font-bold text-3xl">Tasks for This Year</h1>
            </div>

            {withdrawals.total !== 0 || withdrawalsStarted ? (
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
    const {contributionToOrWithdrawalFromSavingsPortfolio} = props
    const withdrawalText =
      contributionToOrWithdrawalFromSavingsPortfolio.type === 'withdrawal'
        ? `Withdraw ${formatCurrency(
            contributionToOrWithdrawalFromSavingsPortfolio.withdrawal
          )} from `
        : `Contribute ${formatCurrency(
            Math.abs(
              contributionToOrWithdrawalFromSavingsPortfolio.contribution
            )
          )} to`

    return (
      <div className={className}>
        <div className="mt-2 p-base">
          <p className="mb-2">
            {withdrawalText} your portfolio and rebalance your remaining
            portfolio of <_Value>{props.afterWithdrawals.balance}</_Value> to
            the following asset allocation:
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
    const {
      withdrawals,
      contributionToOrWithdrawalFromSavingsPortfolio,
      withdrawalsStarted,
    } = props
    return (
      <div className={className}>
        {(withdrawals.total !== 0 || withdrawalsStarted) && (
          <>
            <_HowMuchYouHave className="mb-8" {...props} />
            <_HowMuchToSpend className="mb-8" {...props} />
            {withdrawals.total > 0 && (
              <_HowToFundTheSpending className="mb-8" {...props} />
            )}
          </>
        )}
        {(!withdrawalsStarted ||
          contributionToOrWithdrawalFromSavingsPortfolio.type ===
            'contribution') && <_Contribution className="mb-8" {...props} />}
        <_AssetAllocation className="mb-8" {...props} />
      </div>
    )
  }
)

// -------- HOW MUCH YOU HAVE --------
const _HowMuchYouHave = React.memo(
  ({
    withdrawalsStarted,
    contributions,
    className = '',
    ...props
  }: _Props & {className?: string}) => {
    return (
      <div className={className}>
        <_Heading>How Much You Have</_Heading>
        <p className="mb-4 p-base">
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
          this year. This gives you a total of{' '}
          <_Value className="">{props.afterContributions.balance}</_Value> to
          split between spending for this year and saving for the future.
        </p>
      </div>
    )
  }
)

// -------- HOW MUCH TO SPEND --------
const _HowMuchToSpend = React.memo(
  ({withdrawals, className = '', ...props}: _Props & {className?: string}) => {
    const needsBreakdown =
      withdrawals.essentialByEntry.length > 0 ||
      withdrawals.discretionaryByEntry.length > 0
    const mergeBreakdown = props.strategy === 'SWR'
    return (
      <div className={className}>
        <_Heading>How Much To Spend</_Heading>
        <p className="p-base">
          Out of your <_Value>{props.start.balance}</_Value>, you plan to spend{' '}
          <_Value className="">{withdrawals.total}</_Value> this year
          {needsBreakdown
            ? `,
          broken down as follows:`
            : '.'}
        </p>
        {needsBreakdown && (
          <div
            className="grid justify-start gap-x-10 mt-2 p-base"
            style={{grid: 'auto/auto auto'}}
          >
            <h2 className="">Regular</h2>
            <h2 className="text-right">
              <_Value>{withdrawals.regular}</_Value>
            </h2>
            {mergeBreakdown ? (
              <>
                <h2 className="">Extra</h2>
                <h2></h2>
                {withdrawals.essentialByEntry.map(({id, label, amount}) => (
                  <React.Fragment key={id}>
                    <h2 className="ml-8">{label ?? '<no label>'}</h2>
                    <h2 className="text-right">
                      <_Value>{amount}</_Value>
                    </h2>
                  </React.Fragment>
                ))}
                {withdrawals.discretionaryByEntry.map(({id, label, amount}) => (
                  <React.Fragment key={id}>
                    <h2 className="ml-8">{label ?? '<no label>'}</h2>
                    <h2 className="text-right">
                      <_Value>{amount}</_Value>
                    </h2>
                  </React.Fragment>
                ))}
              </>
            ) : (
              <>
                {withdrawals.essentialByEntry.length > 0 && (
                  <>
                    <h2 className="">Extra - Essential</h2>
                    <h2></h2>
                    {withdrawals.essentialByEntry.map(({id, label, amount}) => (
                      <React.Fragment key={id}>
                        <h2 className="ml-8">{label ?? '<no label>'}</h2>
                        <h2 className="text-right">
                          <_Value>{amount}</_Value>
                        </h2>
                      </React.Fragment>
                    ))}
                  </>
                )}
                {withdrawals.discretionaryByEntry.length > 0 && (
                  <>
                    <h2 className="">Extra - Discretionary</h2>
                    <h2></h2>
                    {withdrawals.discretionaryByEntry.map(
                      ({id, label, amount}) => (
                        <React.Fragment key={id}>
                          <h2 className="ml-8">{label ?? '<no label>'}</h2>
                          <h2 className="text-right">
                            <_Value>{amount}</_Value>
                          </h2>
                        </React.Fragment>
                      )
                    )}
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
    )
  }
)

// -------- HOW TO FUND THE SPENDING --------
const _HowToFundTheSpending = React.memo(
  ({
    withdrawals,
    withdrawalsStarted,
    className = '',
  }: _Props & {className?: string}) => {
    return (
      <div className={className}>
        <_Heading>How To Fund The Spending</_Heading>
        <p className="mb-3 p-base">
          Fund your spending of <_Value>{withdrawals.total}</_Value> from the
          following sources:{' '}
        </p>
        <div
          className="grid  justify-start gap-x-6 p-base"
          style={{grid: 'auto/auto auto'}}
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
  }
)

// -------- CONTRIBUTION --------
const _Contribution = React.memo(
  ({
    contributions,
    contributionToOrWithdrawalFromSavingsPortfolio,
    withdrawalsStarted,
    className = '',
  }: _Props & {className?: string}) => {
    return (
      <div className={className}>
        <_Heading>How Much To Contribute To Your Portfolio</_Heading>
        {contributionToOrWithdrawalFromSavingsPortfolio.type ===
        'contribution' ? (
          contributions.toWithdrawal > 0 ? (
            <p className="mb-3 p-base">
              Your{' '}
              {withdrawalsStarted ? 'retirement income' : 'planned savings'} for
              this year is <_Value>{contributions.total}</_Value> You used{' '}
              <_Value>{contributions.toWithdrawal}</_Value> of this to fund
              spending for this year. Contribute the remaining{' '}
              <_Value className="">{contributions.toSavingsPortfolio}</_Value>{' '}
              to your portfolio.{' '}
            </p>
          ) : (
            <p className="mb-3 p-base">
              Your{' '}
              {withdrawalsStarted ? 'retirement income' : 'planned savings'} for
              this year is <_Value>{contributions.total}</_Value>. Contribute
              all <_Value className="">{contributions.total}</_Value> to your
              portfolio.{' '}
            </p>
          )
        ) : (
          <p className="mb-3 p-base">
            You do not have a contribution to your portfolio this year.
          </p>
        )}
      </div>
    )
  }
)

// -------- ASSET ALLOCATION --------
const _AssetAllocation = React.memo(
  ({
    className = '',

    ...props
  }: _Props & {className?: string}) => {
    const {contributionToOrWithdrawalFromSavingsPortfolio} = props
    return (
      <div className={className}>
        <_Heading>What Asset Allocation To Use</_Heading>
        <div className="mb-3 p-base">
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
          {props.strategy === 'TPAW' && <p className="mt-3">
            <span className="bg-gray-300 px-2 rounded-lg ">Note</span>{' '}
            {`This is the asset allocation for your savings portfolio and will typically 
            be different from the asset allocation for the total portfolio that you entered in the "${paramsInputLabel(
              'stock-allocation'
            )} section." `}
            <Link href="learn/future-savings-and-retirement-income">
              <a className="underline">Learn more.</a>
            </Link>
          </p>}
        </div>
      </div>
    )
  }
)

const _AllocationTable = React.memo(
  ({className = '', ...props}: _Props & {className?: string}) => {
    const {allocation} = props.afterWithdrawals
    const stocks = props.afterWithdrawals.balance * allocation.stocks
    const bonds = props.afterWithdrawals.balance - stocks
    return (
      <div
        className={`${className} grid justify-start gap-x-16`}
        style={{grid: 'auto/auto auto'}}
      >
        <h2 className="">Stocks - {formatPercentage(0)(allocation.stocks)}</h2>
        <h2 className="text-right">
          <_Value>{stocks}</_Value>
        </h2>
        <h2 className="">
          Bonds - {formatPercentage(0)(1 - allocation.stocks)}
        </h2>
        <h2 className="text-right">
          <_Value>{bonds}</_Value>
        </h2>
      </div>
    )
  }
)

const _Heading = React.memo(({children}: {children: ReactNode}) => (
  <h2 className="font-bold text-lg mb-1">{children}</h2>
))

const _Value = React.memo(
  ({children, className}: {children: number; className?: string}) => (
    <span className={className}>{formatCurrency(children)}</span>
  )
)
