import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { Contentful } from '../../../../Utils/Contentful'
import { errorToast } from '../../../../Utils/CustomToasts'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { ToggleSwitch } from '../../../Common/Inputs/ToggleSwitch'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
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
    const { planParams, updatePlanParams, tpawResult } = useSimulation()
    // planParamsExt from result.
    const { asMFN, withdrawalStartMonth } = tpawResult.planParamsExt
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
      planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling ??
        _roundUp(minWithdrawal + (maxWithdrawal - minWithdrawal) / 2, 1000),
    )

    const handleAmount = (amount: number | null) => {
      if (
        amount ===
        planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling
      )
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const floor =
          planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
        if (floor !== null && amount < floor) {
          errorToast('Spending ceiling cannot be lower that spending floor.')
          return floor
        }
        return amount
      })()

      if (amount !== null) setEntryOnEnabled(amount)
      updatePlanParams('setSpendingCeiling', amount)
    }

    const value =
      planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Ceiling</h2>
        <div className="">
          <Contentful.RichText
            body={content.ceiling[planParams.advanced.strategy]}
            p=" p-base"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Ceiling</Switch.Label>
              <ToggleSwitch
                className=""
                checked={value !== null}
                setChecked={(enabled) =>
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
    const { planParams, updatePlanParams, tpawResult } = useSimulation()
    // planParamsExt from result.
    const { asMFN, withdrawalStartMonth } = tpawResult.planParamsExt
    const withdrawalStartAsYFN = asMFN(withdrawalStartMonth)

    const firstWithdrawalOfMinPercentile =
      tpawResult.savingsPortfolio.withdrawals.total
        .byPercentileByMonthsFromNow[0].data[withdrawalStartAsYFN]

    const [entryOnEnabled, setEntryOnEnabled] = useState(
      planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor ??
        Math.min(
          _roundUp(firstWithdrawalOfMinPercentile, 500),
          planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling ??
            Number.MAX_SAFE_INTEGER,
        ),
    )

    const handleAmount = (amount: number | null) => {
      const clone = _.cloneDeep(planParams)
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
      updatePlanParams('setSpendingFloor', amount)
    }

    const value =
      planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Floor</h2>
        <div className="">
          <Contentful.RichText
            body={content.floor[planParams.advanced.strategy]}
            p=" p-base mt-2"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Floor</Switch.Label>
              <ToggleSwitch
                className=""
                checked={value !== null}
                setChecked={(enabled) =>
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

export const PlanInputSpendingCeilingAndFloorSummary = React.memo(() => {
  const { planParams } = useSimulation()
  const { monthlySpendingCeiling, monthlySpendingFloor } =
    planParams.adjustmentsToSpending.tpawAndSPAW
  return (
    <>
      <h2>
        Ceiling:{' '}
        {monthlySpendingCeiling
          ? `${formatCurrency(monthlySpendingCeiling)} per month`
          : 'None'}
      </h2>
      <h2>
        Floor:{' '}
        {monthlySpendingFloor
          ? `${formatCurrency(monthlySpendingFloor)} per month`
          : 'None'}
      </h2>
    </>
  )
})