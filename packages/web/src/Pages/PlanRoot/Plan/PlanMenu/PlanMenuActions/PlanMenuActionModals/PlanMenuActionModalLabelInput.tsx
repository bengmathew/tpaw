import { API } from '@tpaw/common'
import clix from 'clsx'
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
    children,
  }: {
    title: string
    initialLabel: string
    buttonLabel: string
    onCancel: () => void
    onAction: (label: string) => void
    isRunning: boolean
    children?: React.ReactNode
  }) => {
    const [labelUntrimmed, setLabelUntrimmed] = useState(initialLabel ?? '')

    const [validating, setValidating] = useState(false)
    const isValid = useMemo(
      () => !API.UserPlanCreate.parts.label(labelUntrimmed.trim()).error,
      [labelUntrimmed],
    )
    const handleAction = () => {
      if (!isValid) {
        setValidating(true)
      } else {
        onAction(labelUntrimmed.trim())
      }
    }
    const [inputElement, setInputElement] = useState<HTMLInputElement | null>(
      null,
    )
    useEffect(() => inputElement?.focus(), [inputElement])

    return (
      <>
        <h2 className=" dialog-heading">{title}</h2>
        <div className=" dialog-content-div">
          <div className="flex items-center gap-x-4">
            <h2 className="font-bold">Label</h2>
            <input
              ref={setInputElement}
              type="text"
              disabled={isRunning}
              className={` border-2 bg-gray-200 disabled:lighten-2 max-w-[300x]
            ${validating && !isValid ? 'border-red-400' : 'border-gray-200'}
            px-2 py-1.5 rounded-lg w-full `}
              value={labelUntrimmed}
              onChange={(e) => setLabelUntrimmed(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAction()
              }}
            />
          </div>
          {children}
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
            <h2 className={clix(isRunning && 'opacity-0')}>{buttonLabel}</h2>
            {isRunning && <Spinner size="text-xl" />}
          </button>
        </div>
      </>
    )
  },
)
