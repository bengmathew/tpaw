import { fGet } from '@tpaw/common'
import isMobile from 'is-mobile'
import React, { ReactNode, useLayoutEffect, useRef, useState } from 'react'
import ReactDOM from 'react-dom'
import { applyOriginToHTMLElement, Size } from '../../../Utils/Geometry'
import { useWindowSize } from '../../App/WithWindowSize'

// FEATURE: Should probably deprecate ContextMenu in favor of this.

const duration = 300
// const scale = 0.95

export const BasicMenu = React.memo(
  ({
    children,
    align,
  }: {
    children: [ReactNode, (closeMenu: () => void) => ReactNode]
    align: 'left' | 'right'
  }) => {
    const windowSize = useWindowSize()

    const referenceElementRef = useRef<HTMLButtonElement | null>(null)
    const popperElementRef = useRef<HTMLDivElement | null>(null)
    const [size, setSize] = useState<Size | null>(null)
    const [show, setShow] = useState(false)

    useLayoutEffect(() => {
      const observer = new ResizeObserver(() =>
        setSize(fGet(popperElementRef.current).getBoundingClientRect()),
      )
      observer.observe(fGet(popperElementRef.current))
      return () => observer.disconnect()
    }, [])

    const handleShow = () => {
      setShow(true)
      const { width, height } = fGet(size)
      const position = fGet(referenceElementRef.current).getBoundingClientRect()
      const origin = {
        y: Math.min(position.top, windowSize.height - height - 20),
        x: isMobile()
          ? 0
          : align === 'left'
          ? Math.min(position.left, windowSize.width - 10 - width)
          : Math.max(position.right - width, 10),
      }
      applyOriginToHTMLElement(origin, fGet(popperElementRef.current))
    }
    const [opacity0AtTransitionEnd, setOpacity0AtTransitionEnd] = useState(true)
    const invisible = !show && opacity0AtTransitionEnd

    return (
      <>
        <button ref={referenceElementRef} onClick={handleShow}>
          {children[0]}
        </button>
        {ReactDOM.createPortal(
          <div
            className=" page fixed inset-0"
            style={{
              visibility: invisible ? 'hidden' : 'visible',
              transitionProperty: 'opacity',
              transitionDuration: `${duration}ms`,
              opacity: show ? '1' : '0',
            }}
            onTransitionEnd={() => setOpacity0AtTransitionEnd(!show)}
          >
            <div
              className="fixed inset-0 bg-black opacity-70"
              onClick={() => setShow(false)}
            />
            <div
              className={`flex absolute flex-col  rounded-xl   bg-planBG`}
              ref={popperElementRef}
              style={{
                transitionProperty: 'transform',
                transitionDuration: `${duration}ms`,
                transform: `translateY(${show ? '0' : '-10px'})`,
                boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
              }}
            >
              {children[1](() => setShow(false))}
            </div>
          </div>,
          window.document.body,
        )}
      </>
    )
  },
)
