import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { errorToast } from '../../../Utils/CustomToasts'
import { paddingCSSStyle } from '../../../Utils/Geometry'
import { fGet } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../Common/Inputs/SmartDeltaFnForAmountInput'
import { ToggleSwitch } from '../../Common/Inputs/ToggleSwitch'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputSpendingCeilingAndFloor = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <>
          <_SpendingCeilingCard className="" props={props} />
          <_SpendingFloorCard className="mt-8" props={props} />
        </>
      </PlanInputBody>
    )
  },
)

export const _SpendingCeilingCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const content = usePlanContent()['spending-ceiling-and-floor']
    const { params, setPlanParams, tpawResult } = useSimulation()
    // paramsExt from result.
    const { asMFN, withdrawalStartMonth } = tpawResult.paramsExt
    const withdrawalStartAsMFN = asMFN(withdrawalStartMonth)

    const { minWithdrawal, maxWithdrawal } = useMemo(() => {
      const last = fGet(
        _.last(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByMonthsFromNow,
        ),
      ).data
      const first = fGet(
        _.first(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByMonthsFromNow,
        ),
      ).data
      const maxWithdrawal = Math.max(...last.slice(withdrawalStartAsMFN))
      const minWithdrawal = Math.min(...first.slice(withdrawalStartAsMFN))
      return { minWithdrawal, maxWithdrawal }
    }, [tpawResult, withdrawalStartAsMFN])

    const [entryOnEnabled, setEntryOnEnabled] = useState(
      params.plan.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling ??
        _roundUp(minWithdrawal + (maxWithdrawal - minWithdrawal) / 2, 1000),
    )

    const handleAmount = (amount: number | null) => {
      const clone = _.cloneDeep(params.plan)
      if (
        amount ===
        clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling
      )
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const floor =
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
        if (floor !== null && amount < floor) {
          errorToast('Spending ceiling cannot be lower that spending floor.')
          return floor
        }
        return amount
      })()

      if (amount !== null) setEntryOnEnabled(amount)
      clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling = amount
      setPlanParams(clone)
    }

    const value =
      params.plan.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Ceiling</h2>
        <div className="">
          <Contentful.RichText
            body={content.ceiling[params.plan.advanced.strategy]}
            p=" p-base"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Ceiling</Switch.Label>
              <ToggleSwitch
                className=""
                enabled={value !== null}
                setEnabled={(enabled) =>
                  handleAmount(enabled ? entryOnEnabled : null)
                }
              />
            </div>
          </Switch.Group>
          {value !== null && (
            <div className="mt-4">
              <div className={`flex items-center`}>
                <AmountInput
                  className="text-input w-[100px]"
                  prefix="$"
                  disabled={value === null}
                  value={value ?? entryOnEnabled}
                  onChange={handleAmount}
                  decimals={0}
                  modalLabel="Spending Ceiling"
                />
                <h2 className="pl-3">per month</h2>
                <button
                  className="ml-2 px-3"
                  disabled={value === null}
                  onClick={() =>
                    handleAmount(
                      smartDeltaFnForAmountInput.increment(
                        value ?? entryOnEnabled,
                      ),
                    )
                  }
                >
                  <FontAwesomeIcon icon={faPlus} />
                </button>
                <button
                  className="px-3"
                  disabled={value === null}
                  onClick={() =>
                    handleAmount(
                      smartDeltaFnForAmountInput.decrement(
                        value ?? entryOnEnabled,
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
      </div>
    )
  },
)

export const _SpendingFloorCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const content = usePlanContent()['spending-ceiling-and-floor']
    const { params, setPlanParams, tpawResult } = useSimulation()
    // paramsExt from result.
    const { asMFN, withdrawalStartMonth } = tpawResult.paramsExt
    const withdrawalStartAsYFN = asMFN(withdrawalStartMonth)

    const firstWithdrawalOfMinPercentile =
      tpawResult.savingsPortfolio.withdrawals.total
        .byPercentileByMonthsFromNow[0].data[withdrawalStartAsYFN]

    const [entryOnEnabled, setEntryOnEnabled] = useState(
      params.plan.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor ??
        Math.min(
          _roundUp(firstWithdrawalOfMinPercentile, 500),
          params.plan.adjustmentsToSpending.tpawAndSPAW
            .monthlySpendingCeiling ?? Number.MAX_SAFE_INTEGER,
        ),
    )

    const handleAmount = (amount: number | null) => {
      const clone = _.cloneDeep(params.plan)
      if (
        amount === clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
      )
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const ceiling =
          clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling
        if (ceiling !== null && amount > ceiling) {
          errorToast('Spending floor cannot be higher that spending ceiling.')
          return ceiling
        }
        return amount
      })()

      if (amount !== null) setEntryOnEnabled(amount)
      clone.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor = amount
      setPlanParams(clone)
    }

    const value =
      params.plan.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Floor</h2>
        <div className="">
          <Contentful.RichText
            body={content.floor[params.plan.advanced.strategy]}
            p=" p-base mt-2"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Floor</Switch.Label>
              <ToggleSwitch
                className=""
                enabled={value !== null}
                setEnabled={(enabled) =>
                  handleAmount(enabled ? entryOnEnabled : null)
                }
              />
            </div>
          </Switch.Group>
          {value !== null && (
            <div className="mt-4 flex items-center">
              <AmountInput
                className="text-input w-[100px]"
                prefix="$"
                value={value}
                onChange={handleAmount}
                decimals={0}
                modalLabel={'Spending Floor'}
              />
              <h2 className="pl-3">per month</h2>
              <button
                className="ml-2 px-3"
                onClick={() =>
                  handleAmount(smartDeltaFnForAmountInput.increment(value))
                }
              >
                <FontAwesomeIcon icon={faPlus} />
              </button>
              <button
                className="px-3"
                onClick={() =>
                  handleAmount(smartDeltaFnForAmountInput.decrement(value))
                }
              >
                <FontAwesomeIcon icon={faMinus} />
              </button>
            </div>
          )}
        </div>
      </div>
    )
  },
)

const _roundUp = (x: number, step: number) => {
  const xPlusStep = x + step
  return xPlusStep - (xPlusStep % step)
}
