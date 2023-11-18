import { Menu, Transition } from '@headlessui/react'
import clsx from 'clsx'
import React, {
  CSSProperties,
  ReactNode,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import { fGet, noCase } from '../../../Utils/Utils'
import { useWindowSize } from '../../App/WithWindowSize'

export const ContextMenu2 = React.memo(
  ({
    className,
    style,
    children: [buttonChild, menuItemsChild],
    disabled,
    align,
    onMenuClose: onMenuCloseIn,
    getMarginToWindow = (windowWidthName) =>
      windowWidthName === 'xs' ? 0 : 20,
  }: {
    className?: string
    style?: CSSProperties
    align: 'left' | 'right'
    disabled?: boolean

    children: [
      ReactNode,
      (
        | ((args: {
            onMenuClose: (x: () => void) => void
            close: () => void
          }) => ReactNode)
        | ReactNode
      ),
    ]
    onMenuClose?: () => void
    getMarginToWindow?: (windowWidthName: string) => number
  }) => {
    const { windowSize, windowWidthName } = useWindowSize()
    const marginToWindow = getMarginToWindow(windowWidthName)

    const [referenceElement, setReferenceElement] =
      useState<HTMLElement | null>(null)
    const [outerElement, setOuterElement] = useState<HTMLElement | null>(null)
    const [menuElement, setMenuElement] = useState<HTMLElement | null>(null)

    const [onMenuClose, setOnMenuClose] = useState<{
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
      <Menu>
        {({ open, close }) => (
          <>
            <Menu.Button
              ref={setReferenceElement}
              className={className}
              disabled={disabled}
              style={style}
            >
              {buttonChild}
            </Menu.Button>
            {ReactDOM.createPortal(
              <Transition
                show={open}
                // pointer-events-none was needed to get the onPointerLeave to get
                // called on chart card to control the chart hover when the chart
                // menu was open.
                className={clsx('page fixed inset-0 pointer-events-none')}
              >
                <Transition.Child
                  ref={setOuterElement}
                  className="absolute inset-0 bg-black transition-opacity duration-300 "
                  enterFrom="opacity-0 "
                  enterTo="opacity-50 "
                  leaveFrom="opacity-50 "
                  leaveTo="opacity-0 "
                  afterLeave={() => {
                    if (onMenuClose) {
                      onMenuClose.callback()
                      setOnMenuClose(null)
                    }
                    onMenuCloseIn?.()
                  }}
                />
                <Transition.Child
                  ref={setMenuElement}
                  // pointer-events-auto needed to cancel the pointer-events-none above.
                  className="absolute z-10  duration-300 pointer-events-auto"
                  style={{ transitionProperty: 'transform, opacity' }}
                  enterFrom=" -translate-y-2 opacity-0  "
                  leaveTo=" -translate-y-2 opacity-0  "
                >
                  {/*  The menu items child should
              also have a rounded-lg to get the right shape for the focus ring. */}
                  <div
                    className="  bg-pageBG max-h-[90vh] overflow-y-auto w-full sm:w-auto rounded-lg"
                    style={{ boxShadow: '2px 2px 15px 5px rgba(0,0,0,0.28)' }}
                  >
                    {typeof menuItemsChild === 'function'
                      ? menuItemsChild({
                          onMenuClose: (x) => {
                            setOnMenuClose({ callback: x })
                          },
                          close,
                        })
                      : menuItemsChild}
                  </div>
                </Transition.Child>
              </Transition>,
              window.document.body,
            )}
          </>
        )}
      </Menu>
    )
  },
)
