import { Menu, Transition } from '@headlessui/react'
import React, {
  CSSProperties,
  ReactNode,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import { fGet } from '../../../Utils/Utils'
import { useWindowSize } from '../../App/WithWindowSize'

export const ContextMenu2 = React.memo(
  ({
    className,
    style,
    children: [buttonChild, menuItemsChild],
    disabled,
    align,
    onMenuClose: onMenuCloseIn,
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
  }) => {
    const windowSize = useWindowSize()

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
    ) => {
      if (!referenceElement || !menuElement) return
      const referenceBounds = fGet(referenceElement).getBoundingClientRect()
      const menuBounds = menuElement.getBoundingClientRect()
      if (align === 'left') {
        menuElement.style.left = `${Math.min(
          referenceBounds.left,
          windowSize.width - menuBounds.width,
        )}px`
      } else {
        menuElement.style.left = `${Math.max(
          referenceBounds.right - menuBounds.width,
          0,
        )}px`
      }
      const windowBottom = windowSize.height - 25
      const menuTop = Math.floor(
        Math.min(referenceBounds.top, windowBottom - menuBounds.height),
      )

      menuElement.style.top = `${menuTop}px`
    }
    const handleResizeRef = useRef(handleResize)
    handleResizeRef.current = handleResize
    useLayoutEffect(
      () => handleResizeRef.current(referenceElement, menuElement),
      [referenceElement, menuElement],
    )

    // This catches window resizes.
    useLayoutEffect(() => {
      if (!outerElement) return
      const resizeObserver = new ResizeObserver(() =>
        handleResizeRef.current(referenceElement, menuElement),
      )
      resizeObserver.observe(outerElement)
      return () => resizeObserver.disconnect()
    }, [menuElement, outerElement, referenceElement])

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
              <Transition show={open} className="page fixed inset-0 ">
                <Transition.Child
                  ref={setOuterElement}
                  className="absolute inset-0 bg-black transition-opacity duration-300"
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
                  className="absolute z-10  duration-300 "
                  style={{ transitionProperty: 'transform, opacity' }}
                  enterFrom=" -translate-y-2 opacity-0  "
                  leaveTo=" -translate-y-2 opacity-0  "
                >
                  {/* The p-0.5 allows the focus ring to be seen clearly. The menu items child should
              also have a rounded-lg to get the right shape for the focus ring. */}
                  <div
                    className="  bg-pageBG max-h-[90vh] overflow-y-auto w-full sm:w-auto rounded-lg p-0.5"
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
