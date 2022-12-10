import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faTrash } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams, ValueForYearRange } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from './AmountInput'
import { RealOrNominalInput } from './RealOrNominalInput'
import { smartDeltaFnForAmountInput } from './SmartDeltaFnForAmountInput'
import { YearRangeInput, YearRangeInputProps } from './YearRangeInput'

export const EditValueForYearRange = React.memo(
  ({
    className = '',
    title,
    onDone,
    entries,
    transitionOut,
    setHideInMain,
    onBeforeDelete,
    index,
    allowableRange,
    choices,
    labelPlaceholder,
  }: {
    className?: string
    title: string
    setHideInMain: (visible: boolean) => void
    onDone: () => void
    onBeforeDelete?: (id: number) => void
    transitionOut: (onDone: () => void) => void
    entries: (params: PlanParams) => ValueForYearRange[]
    index: number
    allowableRange: SimpleRange
    choices: YearRangeInputProps['choices']
    labelPlaceholder: string
  }) => {
    const { params, setParams } = useSimulation()
    const setEntry = (editEntry: (entry: ValueForYearRange) => void) => {
      setParams((params) => {
        const clone = _.cloneDeep(params)
        const entryClone = _.cloneDeep(entry)
        editEntry(entryClone)
        entries(clone).splice(index, 1, entryClone)
        return clone
      })
    }
    const entry = entries(params)[index]

    const { label, value } = entry

    const { increment, decrement } = smartDeltaFnForAmountInput
    return (
      <div className={`${className}`}>
        <h2 className="text-lg font-bold text-center">{title}</h2>
        <div className="mt-6">
          <div
            className="grid gap-y-6 gap-x-4 items-start"
            style={{ grid: 'auto auto / auto 1fr ' }}
          >
            <h2 className=" justify-self-end mt-1.5">Label</h2>
            <input
              type="text"
              className="bg-gray-200 px-2 py-1.5 rounded-lg w-full"
              value={label ?? ''}
              onChange={(e) => {
                setEntry((entry) => {
                  const trimmed = e.target.value.trim()
                  // Cannot set it to trimmed value because we cannot have
                  entry.label = trimmed.length === 0 ? null : e.target.value
                })
              }}
              placeholder={labelPlaceholder}
            />
            <h2 className=" justify-self-end mt-1.5">
              Amount <br /> per Year
            </h2>
            <div
              className="grid gap-x-2"
              style={{
                grid: 'auto auto /  auto',
              }}
            >
              <div className="grid" style={{ grid: 'auto / 1fr auto auto' }}>
                <AmountInput
                  className="w-[100%] text-input"
                  prefix="$"
                  value={value}
                  onChange={(value) =>
                    setEntry((entry) => (entry.value = value))
                  }
                  decimals={0}
                  modalLabel={null}
                />
                <button
                  className="ml-3 px-3"
                  onClick={() =>
                    setEntry((entry) => (entry.value = increment(entry.value)))
                  }
                >
                  <FontAwesomeIcon icon={faPlus} />
                </button>
                <button
                  className="px-3"
                  onClick={() =>
                    setEntry((entry) => (entry.value = decrement(entry.value)))
                  }
                >
                  <FontAwesomeIcon icon={faMinus} />
                </button>
              </div>
              <RealOrNominalInput
                className=""
                nominal={entry.nominal}
                onChange={(nominal) =>
                  setEntry((entry) => (entry.nominal = nominal))
                }
              />
            </div>
          </div>
          <YearRangeInput
            className="mt-6"
            value={entry.yearRange}
            setValue={(yearRange) =>
              setEntry((entry) => (entry.yearRange = yearRange))
            }
            range={allowableRange}
            choices={choices}
            modalTextInputOnMobile={false}
          />
        </div>
        <div className="mt-6 flex justify-between">
          <button
            className="btn-dark btn-md"
            onClick={() => {
              setHideInMain(false)
              transitionOut(onDone)
            }}
          >
            Done
          </button>
          <button
            className="text-errorFG pl-4 text-lg"
            onClick={() => {
              setHideInMain(true)
              onBeforeDelete?.(entry.id)
              transitionOut(() => {
                onDone()
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  entries(clone).splice(index, 1)
                  return clone
                })
              })
            }}
          >
            <FontAwesomeIcon icon={faTrash} />
          </button>
        </div>
      </div>
    )
  },
)
