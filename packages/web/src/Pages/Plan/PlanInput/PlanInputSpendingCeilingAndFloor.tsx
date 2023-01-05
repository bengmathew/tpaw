import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { extendPlanParams } from '../../../TPAWSimulator/PlanParamsExt'
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
    const { params, setParams, tpawResult } = useSimulation()
    const { asYFN, withdrawalStartYear } = extendPlanParams(
      tpawResult.args.params.original,
    )
    const withdrawalStartAsYFN = asYFN(withdrawalStartYear)

    const { minWithdrawal, maxWithdrawal } = useMemo(() => {
      const last = fGet(
        _.last(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByYearsFromNow,
        ),
      ).data
      const first = fGet(
        _.first(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByYearsFromNow,
        ),
      ).data
      const maxWithdrawal = Math.max(...last.slice(withdrawalStartAsYFN))
      const minWithdrawal = Math.min(...first.slice(withdrawalStartAsYFN))
      return { minWithdrawal, maxWithdrawal }
    }, [tpawResult, withdrawalStartAsYFN])

    const [entryOnEnabled, setEntryOnEnabled] = useState(
      params.adjustmentsToSpending.tpawAndSPAW.spendingCeiling ??
        _roundUp(minWithdrawal + (maxWithdrawal - minWithdrawal) / 2, 10000),
    )

    const handleAmount = (amount: number | null) => {
      const clone = _.cloneDeep(params)
      if (amount === clone.adjustmentsToSpending.tpawAndSPAW.spendingCeiling)
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const floor = clone.adjustmentsToSpending.tpawAndSPAW.spendingFloor
        if (floor !== null && amount < floor) {
          errorToast('Spending ceiling cannot be lower that spending floor.')
          return floor
        }
        return amount
      })()

      if (amount !== null) setEntryOnEnabled(amount)
      clone.adjustmentsToSpending.tpawAndSPAW.spendingCeiling = amount
      setParams(clone)
    }

    const value = params.adjustmentsToSpending.tpawAndSPAW.spendingCeiling

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Ceiling</h2>
        <div className="">
          <Contentful.RichText
            body={content.ceiling[params.strategy]}
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
            <div className={`mt-4 flex ${value === null ? 'opacity-40' : ''}`}>
              <AmountInput
                className="text-input"
                prefix="$"
                disabled={value === null}
                value={value ?? entryOnEnabled}
                onChange={handleAmount}
                decimals={0}
                modalLabel="Spending Ceiling"
              />
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
    const { params, setParams, tpawResult, highlightPercentiles } =
      useSimulation()
    const { asYFN, withdrawalStartYear } = extendPlanParams(
      tpawResult.args.params.original,
    )
    const withdrawalStartAsYFN = asYFN(withdrawalStartYear)

    const firstWithdrawalOfFirstHighlightPercentile =
      tpawResult.savingsPortfolio.withdrawals.total.byPercentileByYearsFromNow[
        tpawResult.args.percentiles.indexOf(highlightPercentiles[0])
      ].data[withdrawalStartAsYFN]

    const [entryOnEnabled, setEntryOnEnabled] = useState(
      params.adjustmentsToSpending.tpawAndSPAW.spendingFloor ??
        _roundUp(firstWithdrawalOfFirstHighlightPercentile, 10000),
    )

    const handleAmount = (amount: number | null) => {
      const clone = _.cloneDeep(params)
      if (amount === clone.adjustmentsToSpending.tpawAndSPAW.spendingFloor)
        return
      amount = (() => {
        if (amount === null) return null
        if (amount < 0) return 0
        const ceiling = clone.adjustmentsToSpending.tpawAndSPAW.spendingCeiling
        if (ceiling !== null && amount > ceiling) {
          errorToast('Spending floor cannot be higher that spending ceiling.')
          return ceiling
        }
        return amount
      })()

      if (amount !== null) setEntryOnEnabled(amount)
      clone.adjustmentsToSpending.tpawAndSPAW.spendingFloor = amount
      setParams(clone)
    }

    const value = params.adjustmentsToSpending.tpawAndSPAW.spendingFloor

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Spending Floor</h2>
        <div className="">
          <Contentful.RichText
            body={content.floor[params.strategy]}
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
            <div className="mt-4 flex">
              <AmountInput
                className="text-input"
                prefix="$"
                value={value}
                onChange={handleAmount}
                decimals={0}
                modalLabel={'Spending Floor'}
              />
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
