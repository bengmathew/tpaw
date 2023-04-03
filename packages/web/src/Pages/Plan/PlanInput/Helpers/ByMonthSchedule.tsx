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
    allowableMonthRangeAsMFN,
    editProps: editPropsIn,
    heading,
  }: {
    className?: string
    style?: React.CSSProperties
    entries: (params: PlanParams) => ValueForMonthRange[]
    hideEntryId: number | null
    allowableMonthRangeAsMFN: SimpleRange
    heading: string | null
    editProps: null | {
      defaultMonthRange: MonthRange
      addButtonText: string
      onEdit: (entryId: number, isAdd: boolean) => void
    }
  }) => {
    const { params, setPlanParams } = useSimulation()

    const editProps = editPropsIn
      ? {
          ...editPropsIn,
          handleAdd: () => {
            const entryId =
              Math.max(-1, ...entries(params.plan).map((x) => x.id)) + 1
            setPlanParams((plan) => {
              const clone = _.cloneDeep(plan)
              entries(clone).push({
                id: entryId,
                label: null,
                value: 0,
                nominal: false,
                monthRange: editPropsIn.defaultMonthRange,
              })
              return clone
            })
            editPropsIn.onEdit(entryId, true)
          },
        }
      : null
    return (
      <div className={`${className}`} style={style}>
        {heading ? (
          <div className="flex justify-between gap-x-4 items-center h-[40px]">
            <h2 className={'font-bold text-lg'}>{heading}</h2>
            {editProps?.handleAdd && (
              <button
                className="flex items-center justify-center gap-x-2 py-1 pl-2  "
                onClick={editProps.handleAdd}
              >
                <FontAwesomeIcon className="text-2xl" icon={faPlusThin} />
              </button>
            )}
          </div>
        ) : (
          <div className="flex justify-start gap-x-4 items-center  ">
            {editProps?.handleAdd && (
              <button
                className="flex items-center justify-center gap-x-2 py-2 rounded-full border border-gray-200 px-4 "
                onClick={editProps.handleAdd}
              >
                <FontAwesomeIcon className="text-3xl" icon={faPlusThin} />
                {editProps.addButtonText}
              </button>
            )}
          </div>
        )}
        <div className="flex flex-col">
          {entries(params.plan).map(
            (entry, i) =>
              entry.id !== hideEntryId && (
                <_Entry
                  key={entry.id}
                  className="mt-6"
                  rangeAsMFN={allowableMonthRangeAsMFN}
                  entry={entry}
                  onEdit={
                    editProps ? () => editProps.onEdit(entry.id, false) : null
                  }
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
    rangeAsMFN,
    entry,
    onEdit,
  }: {
    className?: string
    rangeAsMFN: SimpleRange
    entry: ValueForMonthRange
    onEdit: (() => void) | null
  }) => {
    return (
      <button
        className={`${className} block text-start border border-gray-200 rounded-2xl p-3  `}
        disabled={onEdit === null}
        onClick={() => onEdit?.()}
      >
        <ValueForMonthRangeDisplay
          className=""
          labelClassName="font-medium"
          entry={entry}
          rangeAsMFN={rangeAsMFN}
          skipLength={false}
        />
      </button>
    )
  },
)
