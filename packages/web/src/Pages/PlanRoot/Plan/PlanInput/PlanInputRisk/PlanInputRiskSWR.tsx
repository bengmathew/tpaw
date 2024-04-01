import {
  faCircle as faCircleRegular,
  faMinus,
  faPlus,
} from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
  assertFalse,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useState } from 'react'
import { PlanParamsNormalized } from '../../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { formatCurrency } from '../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { AmountInput } from '../../../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { usePlanContent } from '../../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputSummaryGlidePath } from '../Helpers/PlanInputSummaryGlidePath'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskSPAWAndSWRStockAllocationCard } from './PlanInputRiskSPAWAndSWRStockAllocation'

export const PlanInputRiskSWR = React.memo(
  ({ props }: { props: PlanInputBodyPassThruProps }) => {
    return (
      <>
        <PlanInputRiskSPAWAndSWRStockAllocationCard
          className="mt-4"
          props={props}
        />
        <_WithdrawalCard className="mt-8" props={props} />
      </>
    )
  },
)

const _WithdrawalCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">Withdrawal</h2>
        <div className="mt-4">
          <p className="p-base">
            Enter how much you plan to withdraw from the portfolio.
          </p>
        </div>
        <_Rate className="mt-6" />
        <_Amount className="mt-6" />
      </div>
    )
  },
)

const _Rate = React.memo(({ className = '' }: { className?: string }) => {
  const {  updatePlanParams, planParamsNorm } = useSimulation()
  const { withdrawal } = planParamsNorm.risk.swr
  const { ages } = planParamsNorm

  const [lastEntry, setLastEntry] = useState(
    withdrawal.type === 'asPercentPerYear'
      ? withdrawal.percentPerYear
      : DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
          ages.simulationMonths.numWithdrawalMonths,
        ),
  )
  useEffect(() => {
    if (withdrawal.type === 'asPercentPerYear') {
      setLastEntry(withdrawal.percentPerYear)
    }
  }, [withdrawal])

  const rateLabel =
    ages.simulationMonths.withdrawalStartMonth.asMFN === 0
      ? {
          normal: 'Percentage of current portfolio balance',
          title: 'Percentage of current portfolio balance',
        }
      : {
          normal: 'Percentage of savings portfolio at retirement',
          title: 'Percentage of Savings Portfolio at Retirement',
        }

  const handleChange = (percentPerYear: number) =>
    updatePlanParams(
      'setSWRWithdrawalAsPercentPerYear',
      _.clamp(percentPerYear, 0, 1),
    )

  return (
    <div className={`${className}`}>
      <button
        className={`w-full text-left flex`}
        disabled={withdrawal.type === 'asPercentPerYear'}
        onClick={() => handleChange(lastEntry)}
      >
        <FontAwesomeIcon
          className="mr-2 mt-1"
          icon={
            withdrawal.type === 'asPercentPerYear'
              ? faCircleSelected
              : faCircleRegular
          }
        />
        <span className="">{rateLabel.normal}</span>
      </button>
      {withdrawal.type === 'asPercentPerYear' && (
        <div
          className="inline-grid items-stretch ml-5 mt-4"
          style={{ grid: 'auto/auto auto auto auto' }}
        >
          <AmountInput
            className="text-input w-[60px]"
            value={withdrawal.percentPerYear * 100}
            onChange={(percent) => handleChange(percent / 100)}
            suffix="%"
            decimals={1}
            modalLabel={rateLabel.title}
          />
          <div className="flex items-center">
            <h2 className="ml-2">per year </h2>
          </div>
          <button
            className="ml-2 px-3"
            onClick={() => handleChange(withdrawal.percentPerYear + 0.001)}
          >
            <FontAwesomeIcon icon={faPlus} />
          </button>
          <button
            className="px-3"
            onClick={() => handleChange(withdrawal.percentPerYear - 0.001)}
          >
            <FontAwesomeIcon icon={faMinus} />
          </button>
        </div>
      )}
    </div>
  )
})
const _Amount = React.memo(({ className = '' }: { className?: string }) => {
  const { planParamsNorm, updatePlanParams, simulationResult } = useSimulation()
  const { ages } = planParamsNorm
  const { withdrawal } = planParamsNorm.risk.swr

  const [lastEntry, setLastEntry] = useState(
    _.round(
      fGet(
        simulationResult.savingsPortfolio.withdrawals.regular.byPercentileByMonthsFromNow.find(
          (x) => x.percentile === 50,
        ),
      ).data[ages.simulationMonths.withdrawalStartMonth.asMFN],
      -2,
    ),
  )
  useEffect(() => {
    if (withdrawal.type === 'asAmountPerMonth') {
      setLastEntry(withdrawal.amountPerMonth)
    }
  }, [withdrawal])

  const handleChange = (amountPerMonth: number) =>
    updatePlanParams('setSWRWithdrawalAsAmountPerMonth', amountPerMonth)
  return (
    <div className={`${className}`}>
      <button
        className={`w-full text-left`}
        disabled={withdrawal.type === 'asAmountPerMonth'}
        onClick={() => handleChange(lastEntry)}
      >
        <FontAwesomeIcon
          className="mr-2"
          icon={
            withdrawal.type === 'asAmountPerMonth'
              ? faCircleSelected
              : faCircleRegular
          }
        />{' '}
        Amount
      </button>
      {withdrawal.type === 'asAmountPerMonth' && (
        <div className="ml-5">
          <div
            className="inline-grid items-stretch mt-4"
            style={{ grid: 'auto/auto auto auto auto' }}
          >
            <AmountInput
              className="text-input w-[120px]"
              value={withdrawal.amountPerMonth}
              onChange={(amount) => handleChange(amount)}
              prefix="$"
              decimals={0}
              modalLabel="Withdrawal Amount"
            />
            <div className="flex items-center">
              <h2 className="ml-2">per month </h2>
            </div>
            <button
              className="ml-2 px-3"
              onClick={() =>
                handleChange(
                  smartDeltaFnForAmountInput.increment(
                    withdrawal.amountPerMonth,
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faPlus} />
            </button>
            <button
              className="px-3"
              onClick={() =>
                handleChange(
                  smartDeltaFnForAmountInput.decrement(
                    withdrawal.amountPerMonth,
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faMinus} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
})

export const PlanInputRiskSWRSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    const { risk } = planParamsNorm

    return (
      <>
        <h2>Stock Allocation</h2>
        <div className="ml-4">
          <PlanInputSummaryGlidePath
            className=""
            normValue={risk.spawAndSWR.allocation}
          />
        </div>
        <h2>Withdrawal</h2>
        <h2 className="ml-4">
          {risk.swr.withdrawal.type === 'asPercentPerYear'
            ? `${formatPercentage(1)(risk.swr.withdrawal.percentPerYear)} of ${
                planParamsNorm.ages.simulationMonths.withdrawalStartMonth
                  .asMFN === 0
                  ? 'current portfolio balance'
                  : 'savings portfolio at retirement'
              }`
            : risk.swr.withdrawal.type === 'asAmountPerMonth'
              ? `${formatCurrency(
                  risk.swr.withdrawal.amountPerMonth,
                )} per month`
              : risk.swr.withdrawal.type === 'default'
                ? // Default should have been changed to asPercent if we are showing this.
                  assertFalse()
                : noCase(risk.swr.withdrawal)}
        </h2>
      </>
    )
  },
)
