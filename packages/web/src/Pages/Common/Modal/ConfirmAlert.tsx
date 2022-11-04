import _ from 'lodash'
import React, { ReactNode } from 'react'
import { ModalBase } from './ModalBase'
export const ConfirmAlert = React.memo(
  ({
    title,
    children,
    isWarningBG = false,
    isWarningTitle = false,
    isWarningContent = false,
    onCancel,
    option1,
    option2,
  }: {
    title?: string 
    children: React.ReactNode | React.ReactNode[]
    isWarningBG?: boolean
    isWarningTitle?: boolean
    isWarningContent?: boolean
    option1: {
      onBeforeClose?: (close: () => void) => void
      onClose: () => void
      label: Exclude<ReactNode, undefined>
      isWarning?: boolean
    }
    option2?: {
      onOption2: () => void
      label: Exclude<ReactNode, undefined>
      isWarning?: boolean
    }
    onCancel: (() => void) | null
  }) => {
    return (
      <ModalBase bg={isWarningBG ? 'bg-red-200' : undefined}>
        {(transitionOut) => (
          <>
            {title && (
              <h2
                className={`text-lg font-bold mb-4 
              ${isWarningTitle ? 'text-errorFG' : ''}`}
              >
                {title}
              </h2>
            )}
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
              {option2 && (
                <button
                  className={`btn-md relative ${
                    option2.isWarning ? 'btn-dark-warnBG' : 'btn-dark'
                  }`}
                  onClick={() => transitionOut(option2.onOption2)}
                >
                  {option2.label}
                </button>
              )}
              <button
                className={`btn-md relative ${
                  option1.isWarning ? 'btn-dark-warnBG' : 'btn-dark'
                }`}
                onClick={() =>
                  option1.onBeforeClose
                    ? option1.onBeforeClose(() =>
                        transitionOut(option1.onClose),
                      )
                    : transitionOut(option1.onClose)
                }
              >
                {option1.label}
              </button>
            </div>
          </>
        )}
      </ModalBase>
    )
  },
)
