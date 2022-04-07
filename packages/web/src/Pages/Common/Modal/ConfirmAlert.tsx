import _ from 'lodash'
import React from 'react'
import {ModalBase} from './ModalBase'
export const ConfirmAlert = React.memo(
  ({
    title,
    children,
    confirmText,
    isWarningButton = false,
    isWarningBG = false,
    isWarningTitle = false,
    isWarningContent = false,
    onCancel,
    onConfirm,
  }: {
    title: string
    children: React.ReactNode | React.ReactNode[]
    confirmText: string
    isWarningButton?: boolean
    isWarningBG?: boolean
    isWarningTitle?: boolean
    isWarningContent?: boolean
    onCancel: (() => void) | null
    onConfirm: () => void
  }) => {
    return (
      <ModalBase bg={isWarningBG ? 'bg-red-200' : undefined}>
        {transitionOut => (
          <>
            <h2
              className={`text-lg font-bold mb-4 
              ${isWarningTitle ? 'text-errorFG' : ''}`}
            >
              {title}
            </h2>
            {_.flatten([children]).map((x, i) => (
              <div
                key={i}
                className={`mt-2 p-base  
                `}
              >
                <span className={`${isWarningContent ? 'text-red-500' : ''}`}>
                  {x}
                </span>
              </div>
            ))}
            <div className="flex justify-end mt-4 gap-x-4">
              {onCancel && (
                <button
                  className="btn-md btn-none "
                  onClick={() => transitionOut(onCancel)}
                >
                  Cancel
                </button>
              )}
              <button
                className={`btn-md relative ${
                  isWarningButton ? 'btn-dark-warnBG' : 'btn-dark'
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
