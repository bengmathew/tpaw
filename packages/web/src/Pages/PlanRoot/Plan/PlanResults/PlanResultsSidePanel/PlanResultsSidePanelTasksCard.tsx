import { PlanParams, fGet } from '@tpaw/common'
import { default as clix, default as clsx } from 'clsx'
import _ from 'lodash'
import React, { CSSProperties, ReactNode, useState } from 'react'
import { formatCurrency } from '../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { Padding } from '../../../../../Utils/Geometry'
import { getPrecision } from '../../../../../Utils/GetPrecision'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { useSimulationResultInfo } from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { SimulationResult } from '../../../../../Simulator/UseSimulator'
import { FirstMonthSavingsPortfolioDetail } from '../../../../../Simulator/SimulateOnServer/SimulateOnServer'

export const PlanResultsSidePanelTasksCard = React.memo(
  ({
    className,
    style,
    cardPadding,
    layout,
  }: {
    className?: string
    style?: CSSProperties
    cardPadding: Padding
    layout: 'laptop' | 'desktop' | 'mobile'
  }) => {
    const { simulationResult } = useSimulationResultInfo()
    const [showModal, setShowModal] = useState(false)
    const { contributionToOrWithdrawalFromSavingsPortfolio, afterWithdrawals } =
      _getProps(simulationResult)
    const withdrawOrContribute = (() =>
      contributionToOrWithdrawalFromSavingsPortfolio.type === 'withdrawal'
        ? {
            text: 'Withdraw',
            amount: contributionToOrWithdrawalFromSavingsPortfolio.withdrawal,
          }
        : {
            text: 'Contribute',
            amount: contributionToOrWithdrawalFromSavingsPortfolio.contribution,
          })()

    const planColors = usePlanColors()
    return (
      <>
        <button
          className={clix(
            className,
            'block text-start w-full rounded-2xl overflow-hidden ',
          )}
          style={{ ...style, backgroundColor: planColors.results.cardBG }}
          onClick={() => setShowModal(true)}
        >
          <h2 className="font-bold text-[16px] sm:text-[22px]">Tasks</h2>

          <h2 className="font- text-[13px] sm:text-[15px] mt-1">
            {withdrawOrContribute.text}
          </h2>
          <h2 className=" text-[13px]">
            {formatCurrency(withdrawOrContribute.amount)}
          </h2>

          {layout !== 'mobile' && (
            <>
              <h2 className="font- text-[13px] sm:text-[15px] mt-1">
                Rebalance
              </h2>
              <div className="grid" style={{ grid: 'auto/auto auto' }}>
                <h2 className="text-[13px]">Stocks</h2>
                <h2 className="text-[13px] text-right">
                  {formatPercentage(0)(afterWithdrawals.allocation.stocks)}
                </h2>
                <h2 className="text-[13px]">Bonds</h2>
                <h2 className="text-[13px] text-right">
                  {formatPercentage(0)(1 - afterWithdrawals.allocation.stocks)}
                </h2>
              </div>
            </>
          )}
          <div
            className="flex justify-end mt-2"
            style={{
              marginRight: `${-cardPadding.right}px`,
              marginBottom: `${-cardPadding.bottom}px`,
            }}
          >
            <h2
              className={clix('py-0.5 pl-4 pr-4 rounded-tl-lg')}
              style={{
                backgroundColor: planColors.results.darkBG,
                color: planColors.results.fgForDarkBG,
              }}
            >
              More
            </h2>
          </div>
        </button>
        <CenteredModal
          className=" dialog-outer-div"
          show={showModal}
          style={{
            backgroundColor: planColors.results.cardBG,
            color: planColors.results.fg,
          }}
          onOutsideClickOrEscape={() => setShowModal(false)}
        >
          <div className="p-4">
            <h1 className="font-bold text-2xl sm:text-3xl">
              Tasks for This Month
            </h1>
            <TasksForThisMonthContent className="mt-6" />
          </div>
        </CenteredModal>
      </>
    )
  },
)

type _Props = Omit<FirstMonthSavingsPortfolioDetail, 'withdrawals'> & {
  withdrawals: FirstMonthSavingsPortfolioDetail['withdrawals'] & {
    essentialByEntry: { id: string; label: string | null; amount: number }[]
    discretionaryByEntry: { id: string; label: string | null; amount: number }[]
  }
  withdrawalsStarted: boolean
  strategy: PlanParams['advanced']['strategy']
}

const _getProps = (simulationResult: SimulationResult): _Props => {
  const original = simulationResult.firstMonthOfSomeRun
  const { planParamsNormOfResult } = simulationResult

  const withdrawalsStarted =
    planParamsNormOfResult.ages.simulationMonths.withdrawalStartMonth.asMFN ===
    0

  const getWithdrawals = (type: 'essential' | 'discretionary') =>
    simulationResult.savingsPortfolio.withdrawals[type].byId.map(
      ({ id, byPercentileByMonthsFromNow }) => {
        const { label } = fGet(
          planParamsNormOfResult.adjustmentsToSpending.extraSpending[type]
            .filter((x) => x.amountAndTiming.type !== 'inThePast')
            .find((x) => x.id === id),
        )
        const amount = byPercentileByMonthsFromNow[0].data[0]
        return { id, label, amount }
      },
    )

  return {
    ...original,
    withdrawals: {
      ...original.withdrawals,
      essentialByEntry: getWithdrawals('essential'),
      discretionaryByEntry: getWithdrawals('discretionary'),
    },
    withdrawalsStarted,
    strategy: planParamsNormOfResult.advanced.strategy,
  }
}

export const TasksForThisMonthContent = React.memo(
  ({
    className,
    forPrint = false,
  }: {
    className?: string
    forPrint?: boolean
  }) => {
    const { simulationResult } = useSimulationResultInfo()
    const props = _getProps(simulationResult)

    const { withdrawals, withdrawalsStarted } = props

    return (
      <div className={clsx(className, forPrint ? '' : 'font-font2 text-lg')}>
        {withdrawals.total !== 0 || withdrawalsStarted ? (
          <>
            <h2
              className={clix(
                'font-bold text-xl sm:text-2xl mb-4 font-font1 ',
                forPrint && 'text-black',
              )}
            >
              Summary
            </h2>
            <_Summary className="" {...props} forPrint={forPrint} />

            <h2
              className={clix(
                'font-bold text-xl sm:text-2xl mt-10 mb-6  font-font1 ',
                forPrint && 'text-black',
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
  }: _Props & { className: string; forPrint: boolean }) => {
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
  }: _Props & { className: string; forPrint: boolean }) => {
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
  }: _Props & { className?: string; forPrint: boolean }) => {
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>How Much You Have</_Heading>
        <p className="mb-4">
          Your current portfolio is <_Value>{props.start.balance}</_Value> and
          you{' '}
          {!withdrawalsStarted ? (
            <span>
              plan to contribute an additional{' '}
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
  }: _Props & { className?: string; forPrint: boolean }) => {
    const needsBreakdown =
      withdrawals.essentialByEntry.length > 0 ||
      withdrawals.discretionaryByEntry.length > 0
    const mergeBreakdown = props.strategy === 'SWR'
    return (
      <div className={className}>
        <_Heading forPrint={forPrint}>How Much To Spend</_Heading>
        <p className="">
          Out of your <_Value>{props.afterContributions.balance}</_Value>, you
          plan to spend <_Value className="">{withdrawals.total}</_Value> this
          month
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
  }: _Props & { className?: string; forPrint: boolean }) => {
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
  }: _Props & { className?: string; forPrint: boolean }) => {
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
  }: _Props & { className?: string; forPrint: boolean }) => {
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
  ({ className = '', ...props }: _Props & { className?: string }) => {
    const { allocation } = props.afterWithdrawals
    const stocks = props.afterWithdrawals.balance * allocation.stocks
    const bonds = props.afterWithdrawals.balance - stocks
    const stocksFullStr = formatPercentage(1)(allocation.stocks)
    const bondsFullStr = formatPercentage(1)(
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
        {/* <h2></h2> */}
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
  ({ children, forPrint }: { children: ReactNode; forPrint: boolean }) => {
    // Condition ok because forPrint is constant for the lifetime of the
    // component.
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const planColors = !forPrint ? usePlanColors() : null
    return (
      <h2
        className={clix(
          'font-bold text-lg mb-1 font-font1 ',
          forPrint && 'text-black',
        )}
        style={{ color: planColors?.results?.fg ?? undefined }}
      >
        {children}
      </h2>
    )
  },
)

const _Value = React.memo(
  ({ children, className }: { children: number; className?: string }) => (
    <span className={className}>{formatCurrency(children)}</span>
  ),
)
