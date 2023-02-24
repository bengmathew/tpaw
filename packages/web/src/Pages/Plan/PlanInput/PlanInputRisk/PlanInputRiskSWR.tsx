import {
  faCircle as faCircleRegular,
  faMinus,
  faPlus,
} from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT, fGet } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useState } from 'react'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { useSimulation } from '../../../App/WithSimulation'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { usePlanContent } from '../../Plan'
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
    const { params } = useSimulation()
    const content = usePlanContent()['withdrawal']

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
  const { params, setParams, paramsExt } = useSimulation()
  const { withdrawal } = params.risk.swr
  const { withdrawalsStarted } = paramsExt

  const { numRetirementMonths } = paramsExt

  const [lastEntry, setLastEntry] = useState(
    withdrawal.type === 'asPercentPerYear'
      ? withdrawal.percentPerYear
      : DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(numRetirementMonths),
  )
  useEffect(() => {
    if (withdrawal.type === 'asPercentPerYear') {
      setLastEntry(withdrawal.percentPerYear)
    }
  }, [withdrawal])

  const rateLabel = withdrawalsStarted
    ? {
        normal: 'Percentage of current portfolio balance',
        title: 'Percentage of current portfolio balance',
      }
    : {
        normal: 'Percentage of savings portfolio at retirement',
        title: 'Percentage of Savings Portfolio at Retirement',
      }

  const handleChange = (percentPerYear: number) =>
    setParams((params) => {
      const clone = _.cloneDeep(params)
      clone.risk.swr.withdrawal = {
        type: 'asPercentPerYear',
        percentPerYear: _.clamp(percentPerYear, 0, 1),
      }
      return clone
    })
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
  const { params, setParams, tpawResult, paramsExt } = useSimulation()
  const { withdrawal } = params.risk.swr
  const { asMFN, withdrawalStartMonth } = paramsExt

  const [lastEntry, setLastEntry] = useState(
    _.round(
      fGet(
        tpawResult.savingsPortfolio.withdrawals.regular.byPercentileByMonthsFromNow.find(
          (x) => x.percentile === 50,
        ),
      ).data[asMFN(withdrawalStartMonth)],
      -2,
    ),
  )
  useEffect(() => {
    if (params.risk.swr.withdrawal.type === 'asAmountPerMonth') {
      setLastEntry(params.risk.swr.withdrawal.amountPerMonth)
    }
  }, [params])

  const handleChange = (amountPerMonth: number) =>
    setParams((params) => {
      const clone = _.cloneDeep(params)
      clone.risk.swr.withdrawal = { type: 'asAmountPerMonth', amountPerMonth }
      return clone
    })
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
            params.risk.swr.withdrawal.type === 'asAmountPerMonth'
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
