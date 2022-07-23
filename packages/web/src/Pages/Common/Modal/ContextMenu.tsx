import React, {
  ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import {fGet} from '../../../Utils/Utils'
import {useWindowSize} from '../../App/WithWindowSize'

export const ContextMenu = React.memo(
  ({
    referenceElement,
    onClose,
    children,
    darkBG = true,
    align,
  }: {
    referenceElement: HTMLElement
    align: 'left' | 'right'
    onClose: () => void
    children: (onHide: () => void) => ReactElement
    darkBG?: boolean
  }) => {
    const windowSize = useWindowSize()
    const menuRef = useRef<HTMLDivElement | null>(null)

    const handleResize = useCallback(
      (ifNeeded: boolean) => {
        const menu = fGet(menuRef.current)
        const referenceBounds = referenceElement.getBoundingClientRect()
        const menuBounds = menu.getBoundingClientRect()
        if (align === 'left') {
          menu.style.left = `${Math.min(
            referenceBounds.left,
            windowSize.width - menuBounds.width
          )}px`
        } else {
          menu.style.left = `${Math.max(
            referenceBounds.right - menuBounds.width,
            0
          )}px`
        }
        const windowBottom = windowSize.height - 25
        const menuTop = Math.floor(
          Math.min(referenceBounds.top, windowBottom - menuBounds.height)
        )

        if (ifNeeded && Math.round(menuBounds.top) !== menuTop) {
          return
        }
        menu.style.top = `${menuTop}px`
      },
      [align, referenceElement, windowSize]
    )

    useLayoutEffect(() => {
      const resizeObserver = new ResizeObserver(() => {
        handleResize(true)
      })
      resizeObserver.observe(fGet(menuRef.current))
      resizeObserver.observe(referenceElement)
      return () => resizeObserver.disconnect()
    }, [handleResize, referenceElement])

    const [show, setShow] = useState(false)
    useEffect(() => {
      window.setTimeout(() => setShow(true), 0)
    }, [])

    useLayoutEffect(() => handleResize(false), [handleResize])

    return ReactDOM.createPortal(
      <div className="modal-base">
        <div
          className={`fixed inset-0 duration-300 
              ${darkBG || windowSize.width < 640 ? 'bg-black' : ''}
              ${show ? 'opacity-50' : 'opacity-0'}`}
          onClick={e => {
            if (e.target === e.currentTarget) setShow(false)
          }}
          onTransitionEnd={() => {
            if (!show) onClose()
          }}
          style={{transitionProperty: 'opacity'}}
        />
        <div
          className={`absolute overflow-hidden bg-pageBG rounded-lg  duration-300
        flex flex-col 
        ${show ? '' : '  opacity-0'}`}
          ref={menuRef}
          style={{
            boxShadow: '2px 2px 15px 5px rgba(0,0,0,0.28)',
            transitionProperty: 'opacity',
          }}
        >
          {children(() => setShow(false))}
        </div>
      </div>,
      window.document.body
    )
  }
)
