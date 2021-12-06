import {faTrash} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import React, {useState} from 'react'
import {fGet} from '../../../Utils/Utils'
import {ConfirmAlert} from '../Modal/ConfirmAlert'
import {ModalBase} from '../Modal/ModalBase'
import {AmountInput, useAmountInputState} from './AmountInput'
import {ToggleSwitch} from './ToggleSwitch'

export const LabeledValueInput = React.memo(
  ({
    initial,
    heading,
    onCancel,
    onDone,
    onDelete,
  }: {
    initial: {value: number | null; nominal: boolean; label: string | null}
    heading: string
    onCancel: () => void
    onDone: (result: {
      value: number
      nominal: boolean
      label: string | null
    }) => void
    onDelete: (() => void) | null
  }) => {
    const [confirmDelete, setConfirmDelete] = useState(false)
    const [label, setLabel] = useState(initial.label ?? '')
    const amountState = useAmountInputState(initial.value)

    const [nominal, setnominal] = useState(initial.nominal)

    const handleSave = () => {
      const labelTrim = label.trim()
      onDone({
        label: labelTrim.length === 0 ? null : labelTrim,
        value: amountState.amount,
        nominal,
      })
    }
    return (
      <>
        <ModalBase>
          {transitionOut => (
            <>
              <h2 className="text-lg font-bold text-center">{heading}</h2>
              <div className=" p-2 w-[min(100vw-32px,400px)]">
                {/* Dummy button to capture focus on mobile so keyboard won't show */}
                <button className=""></button>
                <div
                  className="grid gap-x-4 items-center"
                  style={{grid: '50px 50px 50px 35px auto/ auto 1fr'}}
                >
                  <h2 className="justify-self-end">Label</h2>
                  <input
                    type="text"
                    className="bg-gray-200 px-2 py-1.5 rounded-lg"
                    value={label}
                    onChange={e => setLabel(e.target.value)}
                  />
                  <div className="justify-self-end">
                    <h2 className="">Amount</h2>
                    <h2
                      className="text-right lighten-2 text-[.75rem] "
                      style={{lineHeight: '.75rem'}}
                    >
                      per year
                    </h2>
                  </div>
                  <AmountInput className="" state={amountState} />
                  <Switch.Group>
                    <Switch.Label className="justify-self-end">
                      Nominal
                    </Switch.Label>
                    <ToggleSwitch
                      className=""
                      enabled={nominal}
                      setEnabled={setnominal}
                    />
                  </Switch.Group>
                </div>
                <div
                  className="grid gap-x-2 items-center"
                  style={{grid: 'auto/auto auto 1fr auto'}}
                >
                  <button
                    className="btn-lg btn-dark  "
                    onClick={() => transitionOut(handleSave)}
                  >
                    Save
                  </button>
                  <button
                    className="btn-lg btn-none"
                    onClick={() => transitionOut(onCancel)}
                  >
                    Cancel
                  </button>
                  <div />
                  {onDelete ? (
                    <button
                      className="px-2 -mr-2 py-2 btn-none text-errorFG justify-end text-xl"
                      onClick={() => setConfirmDelete(true)}
                    >
                      <FontAwesomeIcon icon={faTrash} />
                    </button>
                  ) : (
                    <div />
                  )}
                </div>
              </div>
            </>
          )}
        </ModalBase>
        {confirmDelete && (
          <ConfirmAlert
            title="Confirm Delete"
            confirmText="Delete"
            isWarning
            onCancel={() => setConfirmDelete(false)}
            onConfirm={() => fGet(onDelete)()}
          >
            Are you sure you want to delete this entry?
          </ConfirmAlert>
        )}
      </>
    )
  }
)
