import { Transition, TransitionChild } from '@headlessui/react'
import React, {
  ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import { fGet, noCase } from '../../../Utils/Utils'
import { useSystemInfo } from '../../App/WithSystemInfo'
import clsx from 'clsx'

export const ContextModal = React.memo(
  ({
    children: [buttonChild, menuItemsChild],
    open,
    align,
    afterLeave: afterLeaveIn,
    onOutsideClickOrEscape,
    getMarginToWindow = (windowWidthName) =>
      windowWidthName === 'xs' ? 0 : 20,
  }: {
    align: 'left' | 'right'
    open: boolean
    children: [
      (x: { ref: (x: HTMLElement | null) => void }) => ReactNode,
      (
        | ((args: { afterLeave: (x: () => void) => void }) => ReactNode)
        | ReactNode
      ),
    ]

    onOutsideClickOrEscape: (() => void) | null
    afterLeave?: () => void
    getMarginToWindow?: (windowWidthName: string) => number
  }) => {
    const { windowSize, windowWidthName } = useSystemInfo()
    const marginToWindow = getMarginToWindow(windowWidthName)

    const [referenceElement, setReferenceElement] =
      useState<HTMLElement | null>(null)
    const [outerElement, setOuterElement] = useState<HTMLElement | null>(null)
    const [menuElement, setMenuElement] = useState<HTMLElement | null>(null)

    const [afterLeave, setAfterLeave] = useState<{
      callback: () => void
    } | null>(null)

    const handleResize = (
      referenceElement: HTMLElement | null,
      menuElement: HTMLElement | null,
      padding: number,
    ) => {
      if (!referenceElement || !menuElement) return
      const referenceBounds = fGet(referenceElement).getBoundingClientRect()
      const menuBounds = menuElement.getBoundingClientRect()
      const left =
        align === 'left'
          ? Math.min(
              referenceBounds.left,
              windowSize.width - menuBounds.width - padding,
            )
          : align === 'right'
            ? referenceBounds.right - menuBounds.width
            : noCase(align)
      menuElement.style.left = `${Math.max(left, padding)}px`
      const windowBottom = windowSize.height - 25
      const menuTop = Math.floor(
        Math.min(referenceBounds.top, windowBottom - menuBounds.height),
      )

      menuElement.style.top = `${menuTop}px`
    }
    const handleResizeRef = useRef(handleResize)
    handleResizeRef.current = handleResize
    useLayoutEffect(
      () =>
        handleResizeRef.current(referenceElement, menuElement, marginToWindow),
      [referenceElement, menuElement, marginToWindow],
    )

    const onOutsideClickOrEscapeRef = useRef(onOutsideClickOrEscape)
    onOutsideClickOrEscapeRef.current = onOutsideClickOrEscape
    useEffect(() => {
      const callback = (e: KeyboardEvent) => {
        if (e.key === 'Escape') onOutsideClickOrEscapeRef.current?.()
      }
      window.document.body.addEventListener('keyup', callback)
      return () => window.document.body.removeEventListener('keyup', callback)
    }, [])

    // This catches window resizes.
    useLayoutEffect(() => {
      const resizeObserver = new ResizeObserver(() =>
        handleResizeRef.current(referenceElement, menuElement, marginToWindow),
      )
      if (outerElement) resizeObserver.observe(outerElement)
      if (menuElement) resizeObserver.observe(menuElement)
      return () => resizeObserver.disconnect()
    }, [menuElement, outerElement, referenceElement, marginToWindow])

    return (
      <>
        {buttonChild({ ref: setReferenceElement })}
        {ReactDOM.createPortal(
          <Transition
            show={open}
            as="div"
            // pointer-events-none was needed to get the onPointerLeave to get
            // called on chart card to control the chart hover when the chart
            // menu was open.
            className={clsx(
              'page fixed inset-0',
              !onOutsideClickOrEscape && 'pointer-events-none',
            )}
          >
            <TransitionChild
              as="div"
              ref={setOuterElement}
              className="absolute inset-0 bg-black/50 transition-opacity duration-300 "
              enterFrom="opacity-0 "
              leaveTo="opacity-0 "
              afterLeave={() => {
                if (afterLeave) {
                  afterLeave.callback()
                  setAfterLeave(null)
                }
                afterLeaveIn?.()
              }}
              onClick={onOutsideClickOrEscape ?? undefined}
            />
            <TransitionChild
              as="div"
              ref={setMenuElement}
              // pointer-events-auto needed to cancel the pointer-events-none above.
              className="absolute z-10  duration-300 pointer-events-auto"
              style={{ transitionProperty: 'transform, opacity' }}
              enterFrom=" -translate-y-2 opacity-0  "
              leaveTo=" -translate-y-2 opacity-0  "
            >
              {/*  The menu items child should also have a rounded-lg to get
                   the right shape for the focus ring. */}
              <div
                className="  bg-pageBG max-h-[90vh] overflow-y-auto w-full sm:w-auto rounded-lg"
                style={{ boxShadow: '2px 2px 15px 5px rgba(0,0,0,0.28)' }}
              >
                {typeof menuItemsChild === 'function'
                  ? menuItemsChild({
                      afterLeave: (x) => {
                        setAfterLeave({ callback: x })
                      },
                    })
                  : menuItemsChild}
              </div>
            </TransitionChild>
          </Transition>,
          window.document.body,
        )}
      </>
    )
  },
)
