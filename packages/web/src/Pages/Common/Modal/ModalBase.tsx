import React, {ReactNode, useEffect, useState} from 'react'
import ReactDOM from 'react-dom'

type _State = {show: true} | {show: false; onTransitionEnd: (() => void) | null}
export const ModalBase = React.memo(
  ({
    children,
    onClose,
    bg = 'bg-pageBG',
    maxHeight = '70vh',
    maxWidth = '700px',
  }: {
    children: (
      transitionOut: (onTransitionEnd: () => void) => void
    ) => ReactNode
    onClose?: () => void
    bg?: string
    maxHeight?: string
    maxWidth?: string
  }) => {
    const [state, setState] = useState<_State>({
      show: false,
      onTransitionEnd: null,
    })

    useEffect(() => {
      // Set Timeout makes it more robust in getting the css transition to work.
      // It was not animating in some cases without it.
      window.setTimeout(() => setState({show: true}), 0)
    }, [])
    return ReactDOM.createPortal(
      <div className="modal-base">
        <div
          className={`fixed z-0 inset-0 bg-black 
          ${state.show ? 'opacity-60' : 'opacity-0'}`}
          style={{transition: 'opacity .25s ease'}}
          // Note, if there are multiple tansitions, this will be fired multiple times.
          onTransitionEnd={() => {
            if (!state.show && state.onTransitionEnd) {
              state.onTransitionEnd()
            }
          }}
          onClick={() => {
            if (onClose && state.show) {
              setState({show: false, onTransitionEnd: onClose})
            }
          }}
        />
        <div
          className={`font-font1 relative ${bg} text-pageFG z-10 rounded-lg p-2 sm:p-4 shadow-xl m-2 sm:m-4 overflow-y-scroll
          ${
            state.show
              ? 'opacity-100 transform scale-100'
              : 'opacity-0 transform scale-90'
          }`}
          style={{
            maxHeight,
            maxWidth,
            minWidth: 'min(calc(100vw - 20px), 400px)',
            transition: 'opacity .25s ease, transform .25s ease',
            boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 15px',
          }}
        >
          {children(onTransitionEnd => {
            if (state.show) {
              setState({show: false, onTransitionEnd})
            }
          })}
        </div>
      </div>,
      window.document.body
    )
  }
)
