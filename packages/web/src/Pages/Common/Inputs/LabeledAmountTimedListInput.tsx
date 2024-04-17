import { faPlus as faPlusThin } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  LabeledAmountTimed,
  LabeledAmountTimedLocation,
  generateSmallId,
} from '@tpaw/common'
import clsx from 'clsx'
import React, { useMemo } from 'react'
import { NormalizedLabeledAmountTimed } from '../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { PlanParamsHelperFns } from '../../../UseSimulator/PlanParamsHelperFns'
import { useSimulation } from '../../PlanRoot/PlanRootHelpers/WithSimulation'
import { LabeledAmountTimedDisplay } from '../LabeledAmountTimedDisplay'

export const LabeledAmountTimedListInput = React.memo(
  ({
    className = '',
    style,
    location,
    hideEntryId,
    editProps: editPropsIn,
  }: {
    className?: string
    style?: React.CSSProperties
    location: LabeledAmountTimedLocation
    hideEntryId: string | null
    editProps: {
      defaultAmountAndTiming: LabeledAmountTimed['amountAndTiming']
      addButtonText: string
      onEdit: (entryId: string, isAdd: boolean) => void
    }
  }) => {
    const { planParamsNorm, updatePlanParams } = useSimulation()
    const entries = useMemo(
      () =>
        PlanParamsHelperFns.getLabeledAmountTimedListFromLocation(
          planParamsNorm,
          location,
        ).sort((a, b) => a.sortIndex - b.sortIndex),
      [location, planParamsNorm],
    )

    const editProps = {
      ...editPropsIn,
      handleAdd: () => {
        const sortIndex = Math.max(-1, ...entries.map((x) => x.sortIndex)) + 1
        const entryId = generateSmallId()
        updatePlanParams('addLabeledAmountTimed2', {
          location,
          entryId,
          sortIndex,
          amountAndTiming: editPropsIn.defaultAmountAndTiming,
        })
        editPropsIn.onEdit(entryId, true)
      },
    }
    const handleDelete = (entryId: string) => {
      updatePlanParams('deleteLabeledAmountTimedOrUntimed', {
        location,
        entryId,
      })
    }
    return (
      <div className={`${className}`} style={style}>
        <button
          className="flex items-center justify-center gap-x-2 py-2 rounded-full border border-gray-200 px-4 whitespace-nowrap "
          onClick={editProps.handleAdd}
        >
          <FontAwesomeIcon className="text-3xl" icon={faPlusThin} />
          {editProps.addButtonText}
        </button>
        <div className="flex flex-col">
          {entries.map(
            (entry) =>
              entry.id !== hideEntryId && (
                <_Entry
                  key={entry.id}
                  className="mt-6"
                  entry={entry}
                  onEdit={() => editProps.onEdit(entry.id, false)}
                  onDelete={() => handleDelete(entry.id)}
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
    entry,
    onEdit,
    onDelete,
  }: {
    className?: string
    entry: NormalizedLabeledAmountTimed
    onEdit: () => void
    onDelete: () => void
  }) => {
    return (
      <div className={clsx(className, 'border border-gray-200 rounded-2xl')}>
        {entry.amountAndTiming.type === 'inThePast' ? (
          <div className="p-3">
            <LabeledAmountTimedDisplay
              className=""
              labelClassName="font-medium"
              entry={entry}
            />
            <div className="flex justify-end">
              <button className=" btn2-xs btn2-dark" onClick={onDelete}>
                Remove
              </button>
            </div>
          </div>
        ) : (
          <button
            className={`w-full text-start  p-3 `}
            disabled={onEdit === null}
            onClick={() => onEdit?.()}
          >
            <LabeledAmountTimedDisplay
              className=""
              labelClassName="font-medium"
              entry={entry}
            />
          </button>
        )}
      </div>
    )
  },
)
