import { API } from '@tpaw/common'
import clsx from 'clsx'
import React, { useEffect, useMemo, useState } from 'react'
import { Spinner } from '../../../../../../Utils/View/Spinner'

export const PlanMenuActionModalLabelInput = React.memo(
  ({
    title,
    initialLabel,
    buttonLabel,
    isRunning,
    onCancel,
    onAction,
  }: {
    title: string
    initialLabel: string
    buttonLabel: string
    onCancel: () => void
    onAction: (label: string) => void
    isRunning: boolean
  }) => {
    const [labelRaw, setLabelRaw] = useState(initialLabel ?? '')
    const label = labelRaw.trim()

    const [validating, setValidating] = useState(false)
    const isValid = useMemo(
      () => !API.UserPlanCreate.parts.label(labelRaw).error,
      [labelRaw],
    )
    const handleAction = () => {
      if (!isValid) {
        setValidating(true)
      } else {
        onAction(label)
      }
    }
    const [inputElement, setInputElement] = useState<HTMLInputElement | null>(
      null,
    )
    useEffect(() => inputElement?.focus(), [inputElement])
    
    return (
      <>
        <h2 className=" dialog-heading">{title}</h2>
        <div className=" dialog-content-div flex items-center gap-x-4">
          <h2 className="font-bold">Label</h2>
          <input
            ref={setInputElement}
            type="text"
            disabled={isRunning}
            className={` border-2 bg-gray-200 disabled:lighten-2
${validating && !isValid ? 'border-red-400' : 'border-gray-200'}
px-2 py-1.5 rounded-lg w-full `}
            value={labelRaw}
            onChange={(e) => setLabelRaw(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAction()
            }}
          />
        </div>
        <div className=" dialog-button-div">
          <button
            className=" dialog-button-cancel"
            onClick={onCancel}
            disabled={isRunning}
          >
            Cancel
          </button>
          <button
            className="relative dialog-button-dark"
            disabled={isRunning}
            onClick={handleAction}
          >
            <h2 className={clsx(isRunning && 'opacity-0')}>{buttonLabel}</h2>
            {isRunning && <Spinner size="text-xl" />}
          </button>
        </div>
      </>
    )
  },
)
