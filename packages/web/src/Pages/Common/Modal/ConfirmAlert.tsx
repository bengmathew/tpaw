import React from 'react'
import { ModalBase } from './ModalBase'
export const ConfirmAlert = React.memo(
  ({
    title,
    children,
    confirmText,
    isWarning = false,
    onCancel,
    onConfirm,
  }: {
    title: string
    children: string | string[]
    confirmText: string
    isWarning?: boolean
    onCancel: () => void
    onConfirm: () => void
  }) => {
    return (
      <ModalBase>
        {transitionOut => (
          <>
            <h2 className="text-lg font-bold mb-4">{title}</h2>
            <div className="mt-2">{children} </div>
            <div className="flex justify-end mt-4 gap-x-4">
              <button
                className="btn-md btn-none "
                onClick={() => transitionOut(onCancel)}
              >
                Cancel
              </button>
              <button
                className={`btn-md relative ${
                  isWarning ? 'btn-dark-warnBG' : 'btn-dark'
                }`}
                onClick={() => transitionOut(onConfirm)}
              >
                {confirmText}
              </button>
            </div>
          </>
        )}
      </ModalBase>
    )
  }
)
