import { faPlus as faPlusThin } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MonthRange, ValueForMonthRange, generateSmallId } from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { PlanParamsExtended } from '../../../../../UseSimulator/ExtentPlanParams'
import { SimpleRange } from '../../../../../Utils/SimpleRange'
import { ValueForMonthRangeDisplay } from '../../../../Common/ValueForMonthRangeDisplay'
import { getValueForMonthRangeEntriesByLocation } from '../../../PlanRootHelpers/PlanParamsChangeAction'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'

export const ByMonthSchedule = React.memo(
  ({
    className = '',
    style,
    location,
    hideEntryId,
    allowableMonthRangeAsMFN,
    editProps: editPropsIn,
  }: {
    className?: string
    style?: React.CSSProperties
    location: Parameters<typeof getValueForMonthRangeEntriesByLocation>[1]
    hideEntryId: string | null
    allowableMonthRangeAsMFN: SimpleRange
    editProps: null | {
      defaultMonthRange: MonthRange
      addButtonText: string
      onEdit: (entryId: string, isAdd: boolean) => void
    }
  }) => {
    const { planParams, updatePlanParams, planParamsExt } = useSimulation()
    const entries = useMemo(
      () =>
        _.values(
          getValueForMonthRangeEntriesByLocation(planParams, location),
        ).sort((a, b) => a.sortIndex - b.sortIndex),
      [location, planParams],
    )

    const editProps = editPropsIn
      ? {
          ...editPropsIn,
          handleAdd: () => {
            const sortIndex =
              Math.max(-1, ...entries.map((x) => x.sortIndex)) + 1
            const entryId = generateSmallId()
            updatePlanParams('addValueForMonthRange', {
              location,
              entryId,
              sortIndex,
              monthRange: editPropsIn.defaultMonthRange,
            })
            editPropsIn.onEdit(entryId, true)
          },
        }
      : null
    return (
      <div className={`${className}`} style={style}>
        {editProps?.handleAdd && (
          <button
            className="flex items-center justify-center gap-x-2 py-2 rounded-full border border-gray-200 px-4 whitespace-nowrap "
            onClick={editProps.handleAdd}
          >
            <FontAwesomeIcon className="text-3xl" icon={faPlusThin} />
            {editProps.addButtonText}
          </button>
        )}
        <div className="flex flex-col">
          {entries.map(
            (entry) =>
              entry.id !== hideEntryId && (
                <_Entry
                  key={entry.id}
                  className="mt-6"
                  rangeAsMFN={allowableMonthRangeAsMFN}
                  entry={entry}
                  onEdit={
                    editProps ? () => editProps.onEdit(entry.id, false) : null
                  }
                  planParamsExt={planParamsExt}
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
    planParamsExt,
  }: {
    className?: string
    rangeAsMFN: SimpleRange
    entry: ValueForMonthRange
    onEdit: (() => void) | null
    planParamsExt: PlanParamsExtended
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
          planParamsExt={planParamsExt}
        />
      </button>
    )
  },
)
