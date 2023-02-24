import { faPlus as faPlusThin } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MonthRange, PlanParams, ValueForMonthRange } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { useSimulation } from '../../../App/WithSimulation'
import { ValueForMonthRangeDisplay } from '../../../Common/ValueForMonthRangeDisplay'

export const ByMonthSchedule = React.memo(
  ({
    className = '',
    style,
    entries,
    hideEntryId,
    allowableMonthRange,
    defaultMonthRange,
    heading,
    addButtonText,
    onEdit,
  }: {
    className?: string
    style?: React.CSSProperties
    entries: (params: PlanParams) => ValueForMonthRange[]
    hideEntryId: number | null
    defaultMonthRange: MonthRange
    allowableMonthRange: SimpleRange
    heading: string | null
    addButtonText: string
    onEdit: (entryId: number, isAdd: boolean) => void
  }) => {
    const { params, setParams } = useSimulation()
    const handleAdd = () => {
      const entryId = Math.max(-1, ...entries(params).map((x) => x.id)) + 1
      setParams((params) => {
        const clone = _.cloneDeep(params)
        entries(clone).push({
          id: entryId,
          label: null,
          value: 0,
          nominal: false,
          monthRange: defaultMonthRange,
        })
        return clone
      })
      onEdit(entryId, true)
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
              <FontAwesomeIcon className="text-2xl" icon={faPlusThin} />
            </button>
          </div>
        ) : (
          <div className="flex justify-start gap-x-4 items-center  ">
            <button
              className="flex items-center justify-center gap-x-2 py-2 rounded-full border border-gray-200 px-4 "
              onClick={handleAdd}
            >
              <FontAwesomeIcon className="text-3xl" icon={faPlusThin} />
              {addButtonText}
            </button>
          </div>
        )}
        <div className="flex flex-col">
          {entries(params).map(
            (entry, i) =>
              entry.id !== hideEntryId && (
                <_Entry
                  key={entry.id}
                  className="mt-6"
                  allowableMonthRange={allowableMonthRange}
                  entry={entry}
                  onEdit={() => onEdit(entry.id, false)}
                />
              ),
          )}
        </div>
      </div>
    )
  },
)

const _Entry = React.memo(
  ({
    className = '',
    allowableMonthRange,
    entry,
    onEdit,
  }: {
    className?: string
    allowableMonthRange: SimpleRange
    entry: ValueForMonthRange
    onEdit: () => void
  }) => {
    return (
      <button
        className={`${className} block text-start border border-gray-200 rounded-2xl p-3  `}
        onClick={onEdit}
      >
        <ValueForMonthRangeDisplay
          className=""
          labelClassName="font-medium"
          entry={entry}
          range={allowableMonthRange}
          skipLength={false}
        />
      </button>
    )
  },
)
