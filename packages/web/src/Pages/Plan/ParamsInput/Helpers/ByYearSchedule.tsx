import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faPen} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {
  TPAWParams,
  ValueForYearRange,
  YearRange,
} from '../../../../TPAWSimulator/TPAWParams'
import {formatCurrency} from '../../../../Utils/FormatCurrency'
import {SimpleRange} from '../../../../Utils/SimpleRange'
import {trimAndNullify} from '../../../../Utils/TrimAndNullify'
import {useSimulation} from '../../../App/WithSimulation'
import {smartDeltaFnForAmountInput} from '../../../Common/Inputs/SmartDeltaFnForAmountInput'
import {YearRangeDisplay} from '../../../Common/YearRangeDisplay'

export const ByYearSchedule = React.memo(
  ({
    className = '',
    style,
    entries,
    hideEntry,
    allowableYearRange,
    defaultYearRange,
    heading,
    onEdit,
  }: {
    className?: string
    style?: React.CSSProperties
    entries: (params: TPAWParams) => ValueForYearRange[]
    hideEntry: number | null
    defaultYearRange: YearRange
    allowableYearRange: SimpleRange
    heading: string | null
    onEdit: (index: number, isAdd: boolean) => void
  }) => {
    const {params, setParams} = useSimulation()
    const handleAdd = () => {
      const index = entries(params).length
      setParams(params => {
        const clone = _.cloneDeep(params)
        entries(clone).push({
          id: Math.max(-1, ...entries(clone).map(x => x.id)) + 1,
          label: null,
          value: 0,
          nominal: false,
          yearRange: defaultYearRange,
        })
        return clone
      })
      onEdit(index, true)
    }
    return (
      <div className={`${className}`} style={style}>
        {heading ? (
          <div className="flex justify-between gap-x-4 items-center h-[40px]">
            <h2 className={'font-bold text-lg'}>{heading}</h2>
            <button
              className="flex items-center justify-center gap-x-2 py-1 pl-2  "
              onClick={handleAdd}
            >
              <FontAwesomeIcon className="text-2xl" icon={faPlus} />
            </button>
          </div>
        ) : (
          <div className="flex justify-start gap-x-4 items-center  ">
            <button
              className="flex items-center justify-center gap-x-2   py-2 "
              onClick={handleAdd}
            >
              <FontAwesomeIcon className="text-2xl" icon={faPlus} />
            </button>
          </div>
        )}
        <div className="flex flex-col gap-y-6 mt-2 ">
          {entries(params).map(
            (entry, i) =>
              hideEntry !== i && (
                <_ByYearScheduleEntry
                  key={i}
                  className=""
                  allowableYearRange={allowableYearRange}
                  entry={entry}
                  onEdit={() => onEdit(i, false)}
                  onChangeAmount={amount => {
                    const p = _.cloneDeep(params)
                    entries(p)[i] = {...entry, value: amount}
                    setParams(p)
                  }}
                />
              )
          )}
        </div>
      </div>
    )
  }
)

const _ByYearScheduleEntry = React.memo(
  ({
    className = '',
    allowableYearRange,
    entry,
    onEdit,
    onChangeAmount,
  }: {
    className?: string
    allowableYearRange: SimpleRange
    entry: ValueForYearRange
    onEdit: () => void
    onChangeAmount: (amount: number) => void
  }) => {
    const {increment, decrement} = smartDeltaFnForAmountInput
    const valueStr = formatCurrency(entry.value)
    return (
      <div className={`${className}`}>
        <div className=" flex flex-row justify-between items-start ">
          <div className="">
            <h2 className="font-medium">
              {trimAndNullify(entry.label) ?? '<no label>'}
            </h2>
            <div className="flex justify-between">
              <div className="flex items-stretch">
                <div className="flex flex-row items-center gap-x-2 mr-2">
                  <h2
                    className=""
                    style={{width: `${Math.max(8, valueStr.length) * 9}px`}}
                  >
                    {valueStr}
                  </h2>
                  <h2 className="">{entry.nominal ? 'nominal' : 'real '}</h2>
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
          </div>
          <button className="px-2 -mr-2 py-2" onClick={onEdit}>
            <FontAwesomeIcon className="text-lg" icon={faPen} />
          </button>
        </div>
        <h2 className="">
          <YearRangeDisplay
            value={entry.yearRange}
            range={allowableYearRange}
            lengthInBrackets={false}
          />
        </h2>
      </div>
    )
  }
)
