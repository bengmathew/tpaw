import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faExclamationCircle, faPen} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useState} from 'react'
import {
  checkYearRange,
  TPAWParams,
  ValueForYearRange,
} from '../../../TPAWSimulator/TPAWParams'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {StateObj} from '../../../Utils/UseStateObj'
import {assertFalse, noCase} from '../../../Utils/Utils'
import {ValueForYearRangeInput} from './ValueForYearRangeInput'

type _InputState = {isEdit: false} | {isEdit: true; editIndex: number} | null
export const ByYearSchedule = React.memo(
  ({
    className = '',
    params: paramsObj,
    entries,
    heading,
    addHeading,
    editHeading,
    defaultYearRange,
  }: {
    className?: string
    params: StateObj<TPAWParams>
    entries: (params: TPAWParams) => ValueForYearRange[]
    heading: string | null
    addHeading: string
    editHeading: string
    defaultYearRange: ValueForYearRange['yearRange']
  }) => {
    const {value: params, set: setParams} = paramsObj
    const [input, setInput] = useState<_InputState>(null)

    const updateParamsForAdd = (params: TPAWParams, entry: ValueForYearRange) =>
      entries(params).push(entry)
    const updateParamsForEdit = (
      params: TPAWParams,
      entry: ValueForYearRange,
      i: number
    ) => (entries(params)[i] = entry)

    return (
      <div className={`${className}`}>
        {heading ? (
          <div className="flex justify-between gap-x-4 items-center h-[40px]">
            <h2 className={'font-bold'}>{heading}</h2>
            <button
              className="flex items-center justify-center gap-x-2 py-1 px-2  "
              onClick={() => setInput({isEdit: false})}
            >
              <FontAwesomeIcon className="text-2xl" icon={faPlus} />
            </button>
          </div>
        ) : (
          <div className="flex justify-start gap-x-4 items-center  my-2 ">
            <button
              className="flex items-center justify-center gap-x-2  mt-4  "
              onClick={() => setInput({isEdit: false})}
            >
              <FontAwesomeIcon className="text-2xl" icon={faPlus} />
            </button>
          </div>
        )}
        <div className="flex flex-col gap-y-6 mt-4 ">
          {entries(params).map((entry, i) => (
            <_Entry
              key={i}
              className=""
              params={params}
              entry={entry}
              onEdit={() => setInput({isEdit: true, editIndex: i})}
              onChangeAmount={amount => {
                const p = _.cloneDeep(params)
                updateParamsForEdit(p, {...entry, value: amount}, i)
                setParams(p)
                setInput(null)
              }}
            />
          ))}
        </div>
        {input && (
          <ValueForYearRangeInput
            params={params}
            initial={
              input.isEdit
                ? entries(params)[input.editIndex]
                : {
                    label: '',
                    yearRange: defaultYearRange,
                    value: null,
                    nominal: false,
                  }
            }
            heading={input.isEdit ? editHeading : addHeading}
            onCancel={() => setInput(null)}
            onDone={entry => {
              const p = _.cloneDeep(params)
              input.isEdit
                ? updateParamsForEdit(p, entry, input.editIndex)
                : updateParamsForAdd(p, entry)
              setParams(p)
              setInput(null)
            }}
            onDelete={
              input.isEdit
                ? () => {
                    const p = _.cloneDeep(params)
                    entries(p).splice(input.editIndex, 1)
                    setParams(p)
                    setInput(null)
                  }
                : null
            }
          />
        )}
      </div>
    )
  }
)

const _Entry = React.memo(
  ({
    className = '',
    params,
    entry,
    onEdit,
    onChangeAmount,
  }: {
    className?: string
    params: TPAWParams
    entry: ValueForYearRange
    onEdit: () => void
    onChangeAmount: (amount: number) => void
  }) => {
    const yearRangeCheck = checkYearRange(params, entry.yearRange)
    return (
      <div
        className={`${className} flex flex-row justify-between items-stretch rounded-lg `}
      >
        <div className="">
          <h2 className="font-medium">{entry.label ?? '<no label>'}</h2>
          <div className="flex justify-between">
            <div className="flex items-stretch">
              <div className="flex flex-row items-center gap-x-2 mr-2">
                <h2 className="w-[85px]">{formatCurrency(entry.value)}</h2>
                <h2 className="lighten text-sm">
                  {entry.nominal ? 'nominal' : 'real'}
                </h2>
              </div>
              <button
                className="flex items-center px-2"
                onClick={() => onChangeAmount(increment(entry.value))}
              >
                <FontAwesomeIcon className="text-base" icon={faPlus} />
              </button>
              <button
                className="flex items-center px-2"
                onClick={() => onChangeAmount(decrement(entry.value))}
              >
                <FontAwesomeIcon className="text-base" icon={faMinus} />
              </button>
            </div>
          </div>
          <h2 className="">{_yearRangeToStr(entry.yearRange)}</h2>
          {yearRangeCheck !== 'ok' && (
            <h2 className="  text-red-500">
              <FontAwesomeIcon
                className="mr-2 text-red-500"
                icon={faExclamationCircle}
              />
              <span className="">
                {yearRangeCheck === 'outOfBounds'
                  ? 'ignoring out-of-bound years'
                  : yearRangeCheck === 'startGreaterThanEnd'
                  ? 'ignored - starting year is later than ending year.'
                  : noCase(yearRangeCheck)}
              </span>
            </h2>
          )}
        </div>
        <div className="flex flex-row justify-start items-stretch">
          <button className="px-2 -mr-2" onClick={onEdit}>
            <FontAwesomeIcon className="text-lg" icon={faPen} />
          </button>
        </div>
      </div>
    )
  }
)

const _yearRangeToStr = (yearRange: ValueForYearRange['yearRange']) =>
  yearRange.start === yearRange.end
    ? `at ${_textForSingle(yearRange.start)}`
    : `from ${_textForRange(yearRange.start, 'first')} ${_textForRange(
        yearRange.end,
        'second'
      )}`

const _textForRange = (
  x: ValueForYearRange['yearRange']['start'],
  type: 'first' | 'second'
) => {
  if (x === 'start') {
    return type === 'first' ? 'now' : assertFalse()
  }
  if (x === 'lastWorkingYear') {
    return type === 'first' ? 'last working year' : 'to last working year'
  }
  if (x === 'retirement') {
    return type === 'first' ? 'retirement' : 'to retirement'
  }
  if (x === 'end') {
    return type === 'first' ? assertFalse() : 'onwards'
  }
  if (typeof x === 'number') {
    return type === 'first' ? `age ${x}` : `to ${x}`
  }
  noCase(x)
}

const _textForSingle = (x: ValueForYearRange['yearRange']['start']) => {
  return typeof x === 'number' ? `age ${x}` : x
}

const {increment, decrement} = smartDeltaFn([
  {value: 10000, delta: 1000},
  {value: 30000, delta: 2000},
  {value: 100000, delta: 5000},
  {value: 200000, delta: 10000},
  {value: 500000, delta: 20000},
  {value: 1000000, delta: 50000},
])
