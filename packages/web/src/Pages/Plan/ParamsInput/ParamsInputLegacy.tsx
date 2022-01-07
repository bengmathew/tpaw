import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faPen} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useMemo, useState} from 'react'
import {processTPAWParams, TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {preciseRange} from '../../../Utils/PreciseRange'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {LabeledValueInput} from '../../Common/Inputs/LabeledValueInput'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'

type _InputState = {isEdit: false} | {isEdit: true; editIndex: number} | null

export const ParamsInputLegacy = React.memo(() => {
  const {params} = useSimulation()
  const [key, setKey] = useState(0)
  useEffect(() => setKey(k => k + 1), [params])

  return (
    <div className="">
      <_Inner key={key} />
    </div>
  )
})

// This is needed for AmountInput to pull new values on resetDefault.
const _Inner = React.memo(() => {
  const {params, setParams} = useSimulation()
  const valueState = useAmountInputState(params.legacy.total)

  const [input, setInput] = useState<_InputState>(null)
  const paramsProcessed = useMemo(() => processTPAWParams(params), [params])
  const handleAmount = (amount: number) => {
    if (amount === params.legacy.total) return
    valueState.setAmountStr(`${amount}`)
    const p = _.cloneDeep(params)
    p.legacy.total = amount
    setParams(p)
  }
  return (
    <div className="">
      <p className="">
        How much money would you like to leave as a legacy? Enter a target in
        real dollars.
      </p>
      <div className={`flex items-center gap-x-2 mt-2`}>
        <AmountInput
          className="mt-2"
          state={valueState}
          onAccept={handleAmount}
        />
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleAmount(increment(params.legacy.total))}
        >
          <FontAwesomeIcon className="text-base" icon={faPlus} />
        </button>
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleAmount(decrement(params.legacy.total))}
        >
          <FontAwesomeIcon className="text-base" icon={faMinus} />
        </button>
      </div>

      <p className="mt-6">
        Enter any assets that will apply towards this target like life insurance
        and home equity. This will be subtracted from the target and the
        portfolio will be used to fund the remainder.
      </p>
      <div className="flex justify-start gap-x-4 items-center  my-2 ">
        <button
          className="flex items-center justify-center gap-x-2 py-1 pr-2  "
          onClick={() => setInput({isEdit: false})}
        >
          <FontAwesomeIcon className="text-2xl" icon={faPlus} />
        </button>
      </div>
      <div className="flex flex-col gap-y-6 mt-4 ">
        {params.legacy.external.map((entry, i) => (
          <_Entry
            key={i}
            className=""
            entry={entry}
            onEdit={() => setInput({isEdit: true, editIndex: i})}
          />
        ))}
      </div>

      {params.legacy.external.length > 0 && (
        <div className="mt-8">
          <h2 className="font-medium">Remainder Funded by Portfolio</h2>
          <h2 className="">
            {formatCurrency(paramsProcessed.legacy.target)}{' '}
            <span className="lighten text-sm">real</span>
          </h2>
        </div>
      )}

      {params.legacy.total > 0 && (
        <>
          <h2 className="font-medium mt-4">Stock Allocation for Legacy</h2>
          <SliderInput
            className=""
            height={60}
            pointers={[
              {
                value: params.targetAllocation.legacyPortfolio.stocks,
                type: 'normal',
              },
            ]}
            onChange={([value]) =>
              setParams(params => {
                const p = _.cloneDeep(params)
                p.targetAllocation.legacyPortfolio.stocks = value
                return p
              })
            }
            formatValue={formatPercentage(0)}
            domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
              value: value,
              tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
            }))}
          />
        </>
      )}
      {input && (
        <LabeledValueInput
          initial={
            input.isEdit
              ? params.legacy.external[input.editIndex]
              : {
                  label: '',
                  value: null,
                  nominal: false,
                }
          }
          heading={input.isEdit ? 'Edit Legacy Entry' : 'Add Legacy Entry'}
          onCancel={() => setInput(null)}
          onDone={entry => {
            setParams(params => {
              const p = _.cloneDeep(params)
              input.isEdit
                ? (p.legacy.external[input.editIndex] = entry)
                : p.legacy.external.push(entry)
              return p
            })
            setInput(null)
          }}
          onDelete={
            input.isEdit
              ? () => {
                  const p = _.cloneDeep(params)
                  p.legacy.external.splice(input.editIndex, 1)
                  setParams(p)
                  setInput(null)
                }
              : null
          }
        />
      )}
    </div>
  )
})

const _Entry = React.memo(
  ({
    className = '',
    entry,
    onEdit,
  }: {
    className?: string
    entry: TPAWParams['legacy']['external'][0]
    onEdit: () => void
  }) => (
    <div
      className={`${className} flex flex-row justify-between items-stretch rounded-lg `}
    >
      <div className="">
        <h2 className="font-medium">{entry.label ?? '<no label>'}</h2>
        <div className="flex justify-between">
          <div className="flex items-stretch">
            <div className="flex flex-row items-center gap-x-2 mr-2">
              <h2 className="">{formatCurrency(entry.value)}</h2>
              <h2 className="lighten text-sm">
                {entry.nominal ? 'nominal' : 'real'}
              </h2>
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-row justify-start items-stretch">
        <button className="px-2 -mr-2" onClick={onEdit}>
          <FontAwesomeIcon className="text-lg" icon={faPen} />
        </button>
      </div>
    </div>
  )
)

const {increment, decrement} = smartDeltaFn([
  {value: 1000000, delta: 100000},
  {value: 2000000, delta: 250000},
])
