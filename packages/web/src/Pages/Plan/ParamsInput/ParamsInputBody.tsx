import { Transition } from '@headlessui/react'
import { gsap } from 'gsap'
import {
  default as React,
  ReactElement,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from 'react'
import { assert, fGet } from '../../../Utils/Utils'
import { useWindowSize } from '../../../Utils/WithWindowSize'
import { ModalBase } from '../../Common/Modal/ModalBase'
import { ParamsInputHeading } from './ParamsInput'

const duration = 500

export type ParamsInputBodyProps = Omit<
  React.ComponentProps<typeof ParamsInputBody>,
  'children'
>

export const ParamsInputBody = React.memo(
  ({
    className,
    headingProps,
    children: childrenIn,
  }: {
    className: string
    headingProps: React.ComponentProps<typeof ParamsInputHeading>
    children:
      | ReactElement
      | [
          ReactElement,
          {
            error?: ReactElement
            input?: (
              transitionOut: (onDone: () => void) => void
            ) => ReactElement
          }
        ]
  }) => {
    const {layout} = headingProps
    const [content, children] =
      childrenIn instanceof Array ? childrenIn : ([childrenIn, null] as const)

    const child = {...children, content}

    return layout === 'mobile' ? (
      <_Mobile className={className} headingProps={headingProps}>
        {child}
      </_Mobile>
    ) : (
      <_LaptopAndDesktop
        className={className}
        headingProps={headingProps}
        layout={layout}
      >
        {child}
      </_LaptopAndDesktop>
    )
  }
)

export const _LaptopAndDesktop = React.memo(
  ({
    layout,
    className,
    headingProps,
    children,
  }: {
    layout: 'laptop' | 'desktop'
    className: string
    headingProps: React.ComponentProps<typeof ParamsInputHeading>
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const sizerRef = useRef<HTMLDivElement | null>(null)
    const mainRef = useRef<HTMLDivElement | null>(null)
    const [main, setMain] = useState<HTMLDivElement | null>(null)
    const [input, setInput] = useState<HTMLDivElement | null>(null)
    const [state, setState] = useState<
      {type: 'main'; onDone?: () => void} | {type: 'input'}
    >({type: 'main'})
    const hasInput = children?.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) {
        setState({type: 'input'})
      }
    }, [hasInput])

    const [mainH, setMainH] = useState(0)
    const [inputH, setInputH] = useState(0)
    let height = state.type === 'input' ? inputH : mainH
    const windowSize = useWindowSize()
    if (layout === 'laptop') {
      height = Math.min(windowSize.height * 0.55, height)
    }

    useEffect(() => {
      // To prevent initial resize to 0 and then to height. Initial size will
      // be set in a useLayoutEffect.
      if (height === 0) return
      const sizer = fGet(sizerRef.current)
      if (layout === 'desktop') {
        // Size is set by parent grid.
        sizer.style.height = 'inherit'
      } else if (layout === 'laptop') {
        // Size is set manually here.
        gsap.to(sizer, {height: `${height}px`})
      }
    }, [height, layout])

    useLayoutEffect(() => {
      if (!main) return
      const mainObserver = new ResizeObserver(() => {
        setMainH(main.getBoundingClientRect().height)
      })
      mainObserver.observe(main)
      return () => mainObserver.disconnect()
    }, [main])

    useLayoutEffect(() => {
      if (!input) return
      const inputObserver = new ResizeObserver(() => {
        setInputH(input.getBoundingClientRect().height)
      })
      inputObserver.observe(fGet(input))
      return () => inputObserver.disconnect()
    }, [input])

    useLayoutEffect(() => {
      const height = fGet(mainRef.current).getBoundingClientRect().height
      fGet(sizerRef.current).style.height = `${height}px`
    }, [])

    return (
      <div className={`relative`} ref={sizerRef}>
        {/* Scroll Container. Main and input needs separate scroll containers,
           so they don't interfere with each other's scroll. */}
        <Transition
          // Needed to preserve scroll.
          unmount={false}
          show={state.type === 'main'}
          // Classname should be inside scroll container, because padding should
          // be inside it to place scrollbar at edge.
          className={`absolute inset-0 transition-all overflow-y-scroll  ${className}`}
          enterFrom="opacity-0 -translate-x-4"
          leaveTo="opacity-0 -translate-x-4"
          style={{transitionDuration: `${duration}ms`}}
        >
          <div
            className=""
            ref={main => {
              // Main is needed as state to trigger resizeObserver creation and
              // destruction. But as state, main will not be available in
              // initial useLayoutEffect to set size, so we need mainRef as
              // well.
              mainRef.current = main
              setMain(main)
            }}
          >
            <ParamsInputHeading {...headingProps} />
            <div className="pb-8">{children.content}</div>
            {children?.error && (
              // Cannot set mb-3 and bottom-3 because it leads to scroll.
              <div className="pb-3 sticky bottom-0">
                <div className=" bg-red-100 rounded-lg p-2">
                  {children?.error}
                </div>
              </div>
            )}
          </div>
        </Transition>
        {children?.input && (
          // Scroll container. Main and input needs separate scroll containers,
          // so they don't interfere with each other's scroll.
          <Transition
            show={state.type === 'input'}
            // Classname should be inside scroll container, because padding
            // should be inside it to place scrollbar at edge.
            className={`absolute inset-0  transition-all overflow-y-scroll ${className}`}
            enterFrom="opacity-0 translate-x-4"
            leaveTo="opacity-0 translate-x-4"
            afterLeave={() => {
              assert(state.type === 'main')
              state.onDone?.()
              setState({type: 'main'})
            }}
            style={{transitionDuration: `${duration}ms`}}
          >
            <div className={``} ref={setInput}>
              <div className="pb-8">
                {fGet(children.input)(onDone =>
                  setState({type: 'main', onDone})
                )}
              </div>
            </div>
          </Transition>
        )}
      </div>
    )
  }
)

export const _Mobile = React.memo(
  ({
    className,
    headingProps,
    children,
  }: {
    className: string
    headingProps: React.ComponentProps<typeof ParamsInputHeading>
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const [showInput, setShowInput] = useState(false)
    const hasInput = children.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) setShowInput(true)
    }, [hasInput])
    return (
      <div className={className}>
        <ParamsInputHeading {...headingProps} />
        <div className="pb-8">{children.content}</div>
        {children?.error && (
          <div className="pb-3 sticky bottom-0">
            <div className=" bg-red-100 rounded-lg p-2">{children?.error}</div>
          </div>
        )}
        {showInput && (
          <ModalBase>
            {transitionOut => (
              <div className="px-2 pb-4">
                {fGet(children.input)(onDone =>
                  transitionOut(() => {
                    setShowInput(false)
                    onDone()
                  })
                )}
              </div>
            )}
          </ModalBase>
        )}
      </div>
    )
  }
)
