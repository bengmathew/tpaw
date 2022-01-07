import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import _ from 'lodash'
import React, {useState} from 'react'
import {fGet} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'

export const ParamsInputSpendingCeiling = React.memo(() => {
  const {params, setParams, tpawResult} = useSimulation()

  const [lastEntry, setLastEntry] = useState(() => {
    if (params.spendingCeiling) return params.spendingCeiling
    const targetY =
      (tpawResult.minWithdrawal + tpawResult.maxWithdrawal) / 2 + 10000
    return targetY - (targetY % 10000)
  })
  const valueState = useAmountInputState(lastEntry)

  const handleAmount = (amount: number) => {
    if (amount === params.spendingCeiling) return
    setLastEntry(amount)
    valueState.setAmountStr(`${amount}`)
    const p = _.cloneDeep(params)
    p.spendingCeiling = amount
    setParams(p)
  }

  return (
    <div className="">
      <div className="flex flex-col gap-y-4 items-start">
        <Switch.Group as="div" className="flex gap-x-4 items-center">
          <Switch.Label className="justify-self-end">
            Use a Spending Ceiling
          </Switch.Label>
          <ToggleSwitch
            className=""
            enabled={params.spendingCeiling !== null}
            setEnabled={enabled => {
              const p = _.cloneDeep(params)
              if (!enabled) {
                p.spendingCeiling = null
              } else {
                p.spendingCeiling = lastEntry
              }
              setParams(p)
            }}
          />
        </Switch.Group>
        <div
          className={` mt-2
          ${params.spendingCeiling === null ? ' opacity-30 ' : ''}`}
        >
          <h2 className="">Enter the spending ceiling in real dollars.</h2>
          <div
            className={`flex items-center gap-x-2 mt-2
          ${params.spendingCeiling === null ? ' opacity-30 ' : ''}`}
          >
            <AmountInput
              className={`mr-4 `}
              disabled={params.spendingCeiling === null}
              state={valueState}
              onAccept={handleAmount}
            />
            <button
              className={`flex items-center px-2 `}
              disabled={params.spendingCeiling === null}
              onClick={() => handleAmount(fGet(params.spendingCeiling) + 5000)}
            >
              <FontAwesomeIcon className="text-base" icon={faPlus} />
            </button>
            <button
              className={`flex items-center px-2 `}
              disabled={params.spendingCeiling === null}
              onClick={() =>
                handleAmount(Math.max(0, fGet(params.spendingCeiling) - 5000))
              }
            >
              <FontAwesomeIcon className="text-base" icon={faMinus} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
})
