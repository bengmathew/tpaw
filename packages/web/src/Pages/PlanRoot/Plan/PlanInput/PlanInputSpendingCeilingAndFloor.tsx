import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { Contentful } from '../../../../Utils/Contentful'
import { errorToast } from '../../../../Utils/CustomToasts'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { smartDeltaFnForMonthlyAmountInput } from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { SwitchAsToggle } from '../../../Common/Inputs/SwitchAsToggle'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputSpendingCeilingAndFloor = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParamsNorm, simulationResult } = useSimulation()
    const { initialCeiling, initialFloor } = useMemo(() => {
      const { ages } = planParamsNorm

      const last = fGet(
        _.last(
          simulationResult.savingsPortfolio.withdrawals.total
            .byPercentileByMonthsFromNow,
        ),
      ).data
      const first = fGet(
        _.first(
          simulationResult.savingsPortfolio.withdrawals.total
            .byPercentileByMonthsFromNow,
        ),
      ).data
      const maxWithdrawal = Math.max(
        ...last.slice(ages.simulationMonths.withdrawalStartMonth.asMFN),
      )
      const minWithdrawal = Math.min(
        ...first.slice(ages.simulationMonths.withdrawalStartMonth.asMFN),
      )
      const initialCeiling = smartDeltaFnForMonthlyAmountInput.increment(
        minWithdrawal + (maxWithdrawal - minWithdrawal) / 2,
      )
      const initialFloor = Math.min(
        smartDeltaFnForMonthlyAmountInput.increment(
          minWithdrawal + (maxWithdrawal - minWithdrawal) * 0.05,
        ),
        planParamsNorm.adjustmentsToSpending.tpawAndSPAW
          .monthlySpendingCeiling ?? Number.MAX_SAFE_INTEGER,
      )

      return { initialCeiling, initialFloor }
    }, [planParamsNorm, simulationResult])
    return (
      <PlanInputBody {...props}>
        <>
          <_SpendingCeilingCard
            className=""
            props={props}
            initialCeiling={initialCeiling}
          />
          <_SpendingFloorCard
            className="mt-8"
            props={props}
            initialFloor={initialFloor}
          />
        </>
      </PlanInputBody>
    )
  },
)

export const _SpendingCeilingCard = React.memo(
  ({
    className = '',
    props,
    initialCeiling,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    initialCeiling: number
  }) => {
    const content = usePlanContent()['spending-ceiling-and-floor']
    const { planParamsNorm, updatePlanParams } = useSimulation()

    const [entryOnEnabled, setEntryOnEnabled] = useState(initialCeiling)

    const handleAmount = (amount: number | null) => {
      if (
        amount ===
        planParamsNorm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling
      )
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const floor =
          planParamsNorm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
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
      planParamsNorm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Ceiling</h2>
        <div className="">
          <Contentful.RichText
            body={content.ceiling[planParamsNorm.advanced.strategy]}
            p=" p-base"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Ceiling</Switch.Label>
              <SwitchAsToggle
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
                      smartDeltaFnForMonthlyAmountInput.increment(
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
                      smartDeltaFnForMonthlyAmountInput.decrement(
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
    initialFloor,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    initialFloor: number
  }) => {
    const content = usePlanContent()['spending-ceiling-and-floor']
    const { planParamsNorm, updatePlanParams } = useSimulation()

    const [entryOnEnabled, setEntryOnEnabled] = useState(initialFloor)

    const handleAmount = (amount: number | null) => {
      if (
        amount ===
        planParamsNorm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
      )
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const ceiling =
          planParamsNorm.adjustmentsToSpending.tpawAndSPAW
            .monthlySpendingCeiling
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
      planParamsNorm.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Floor</h2>
        <div className="">
          <Contentful.RichText
            body={content.floor[planParamsNorm.advanced.strategy]}
            p=" p-base mt-2"
          />
        </div>
        <div className="mt-6">
          <Switch.Group>
            <div className="flex items-center gap-x-2">
              <Switch.Label className="">Enable Floor</Switch.Label>
              <SwitchAsToggle
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
                  handleAmount(
                    smartDeltaFnForMonthlyAmountInput.increment(value),
                  )
                }
              >
                <FontAwesomeIcon icon={faPlus} />
              </button>
              <button
                className="px-3"
                onClick={() =>
                  handleAmount(
                    smartDeltaFnForMonthlyAmountInput.decrement(value),
                  )
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

export const PlanInputSpendingCeilingAndFloorSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    const { monthlySpendingCeiling, monthlySpendingFloor } =
      planParamsNorm.adjustmentsToSpending.tpawAndSPAW
    return (
      <>
        <h2>
          Ceiling:{' '}
          {monthlySpendingCeiling !== null
            ? `${formatCurrency(monthlySpendingCeiling)} per month`
            : 'None'}
        </h2>
        <h2>
          Floor:{' '}
          {monthlySpendingFloor !== null
            ? `${formatCurrency(monthlySpendingFloor)} per month`
            : 'None'}
        </h2>
      </>
    )
  },
)
