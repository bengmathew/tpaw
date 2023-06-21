import { Transition } from '@headlessui/react'
import clsx from 'clsx'
import React, { ReactNode, useEffect } from 'react'
import ReactDOM from 'react-dom'

export const CenteredModal = React.memo(
  ({
    show,
    onOutsideClickOrEscape,
    children,
    className,
    onLeave,
  }: {
    show: boolean
    onOutsideClickOrEscape: (() => void) | null
    children: ReactNode
    className?: string
    onLeave?: () => void
  }) => {
    const forEventRefs = React.useRef({ onOutsideClickOrEscape })
    forEventRefs.current = { onOutsideClickOrEscape }
    useEffect(() => {
      const { onOutsideClickOrEscape } = forEventRefs.current
      if (!onOutsideClickOrEscape) return
      const callback = (e: KeyboardEvent) => {
        if (e.key === 'Escape') onOutsideClickOrEscape()
      }
      window.document.body.addEventListener('keyup', callback)
      return () => window.document.body.removeEventListener('keyup', callback)
    }, [])
    return ReactDOM.createPortal(
      <Transition
        show={show}
        appear
        className="page fixed inset-0  flex justify-center items-center"
        afterLeave={onLeave}
      >
        <Transition.Child
          className="absolute inset-0 bg-black/50 transition-opacity duration-300"
          enterFrom="opacity-0 "
          leaveFrom="opacity-50 "
          onClick={onOutsideClickOrEscape ?? undefined}
        />
        <Transition.Child
          className={clsx(
            className,
            'duration-300 bg-pageBG rounded-xl z-10 p-4 max-h-[85vh] overflow-scroll',
          )}
          style={{
            transitionProperty: 'transform, opacity',
            boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 15px',
          }}
          enterFrom=" -translate-y-4 opacity-0  "
          leaveTo=" -translate-y-4 opacity-0"
        >
          {children}
        </Transition.Child>
      </Transition>,
      document.body,
    )
  },
)
