import {gsap} from 'gsap'
import {
  default as React,
  ReactElement,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {Padding, paddingCSS} from '../../../Utils/Geometry'
import {fGet} from '../../../Utils/Utils'
import {Footer} from '../../App/Footer'
import {ModalBase} from '../../Common/Modal/ModalBase'

const duration = 500 / 1000

export type ParamsInputBodyProps = Omit<
  React.ComponentProps<typeof ParamsInputBody>,
  'children'
>

export const ParamsInputBody = React.memo(
  ({
    layout,
    padding,
    cardPadding,
    children: childrenIn,
  }: {
    layout: 'mobile' | 'laptop' | 'desktop'
    padding: Padding
    cardPadding: Padding
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
    const [content, children] =
      childrenIn instanceof Array ? childrenIn : ([childrenIn, null] as const)

    const child = {...children, content}

    return layout === 'mobile' ? (
      <_Mobile padding={padding}>{child}</_Mobile>
    ) : (
      <_LaptopAndDesktop
        padding={padding}
        cardPadding={cardPadding}
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
    padding,
    cardPadding,
    children,
  }: {
    layout: 'laptop' | 'desktop'
    padding: Padding
    cardPadding: Padding
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const [state, setState] = useState<
      {type: 'main'; onDone?: () => void} | {type: 'input'}
    >({type: 'main'})
    const hasInput = children?.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) {
        setState({type: 'input'})
      }
    }, [hasInput])

    const mainScrollRef = useRef<HTMLDivElement | null>(null)
    const inputScrollRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => {
      if (!inputScrollRef.current) return
      const tween = gsap.timeline()
      const toMain = state.type === 'main'
      const mainScroll = fGet(mainScrollRef.current)
      const inputScroll = inputScrollRef.current
      mainScroll.style.display = 'block'
      inputScroll.style.display = 'block'
      tween.to(
        mainScroll,
        {
          opacity: toMain ? 1 : 0,
          x: toMain ? 0 : -16,
          onComplete: () => {
            if (state.type === 'main') {
              state.onDone?.()
              setState({type: 'main'})
            } else {
              mainScroll.style.display = 'none'
            }
          },
          duration,
        },
        0
      )
      tween.fromTo(
        inputScroll,
        toMain ? {opacity: 1, x: 0} : {opacity: 0, x: 16},
        {
          ...(toMain ? {opacity: 0, x: 16} : {opacity: 1, x: 0}),
          duration,
        },
        0
      )
    }, [state])

    return (
      <>
        {/* Scroll Container. Main and input needs separate scroll containers,
           so they don't interfere with each other's scroll. */}
        <div
          className={`absolute inset-0  max-h-full overflow-y-scroll `}
          ref={mainScrollRef}
        >
          <div
            className={`h-full ${layout !== 'laptop' ? 'grid' : ''}`}
            style={{grid: '1fr auto/1fr'}}
          >
            <div
              className="" // Padding should be inside scroll container to place scrollbar at
              // edge and at main to get full height.
              style={{padding: paddingCSS(padding)}}
            >
              <div
                className="bg-cardBG rounded-2xl"
                style={{padding: paddingCSS(cardPadding)}}
              >
                <div className="">{children.content}</div>
              </div>
              {children?.error && (
                // Cannot set mb-3 and bottom-3 because it leads to scroll.
                <div className=" sticky bottom-0 pt-4">
                  <div className=" bg-red-100 rounded-lg p-2">
                    {children?.error}
                  </div>
                </div>
              )}
            </div>
            {layout !== 'laptop' && <Footer />}
          </div>
        </div>
        {children?.input && (
          // Scroll container. Main and input needs separate scroll containers,
          // so they don't interfere with each other's scroll.
          <div
            className={`absolute inset-0  overflow-y-scroll`}
            ref={inputScrollRef}
            // Padding should be inside scroll container to place scrollbar at
            // edge and at input to get full height.
            style={{padding: paddingCSS(padding)}}
          >
            <div
              className={`bg-cardBG rounded-2xl`}
              style={{padding: paddingCSS(cardPadding)}}
            >
              <div className="">
                {fGet(children.input)(onDone =>
                  setState({type: 'main', onDone})
                )}
              </div>
            </div>
          </div>
        )}
      </>
    )
  }
)

export const _Mobile = React.memo(
  ({
    padding,
    children,
  }: {
    padding: Padding
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const [showInput, setShowInput] = useState(false)
    const [showError, setShowError] = useState(false)
    const hasInput = children.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) setShowInput(true)
    }, [hasInput])

    useEffect(() => {
      if (!children.error) setShowError(false)
    }, [children.error])
    return (
      <div
        className="absolute inset-0  max-h-full"
        style={{padding: paddingCSS(padding)}}
      >
        <div className="h-full grid  overflow-y-scroll" style={{grid: '1fr auto/1fr'}}>
          <div className="pb-16">{children.content}</div>
          <div className="-mb-[60px] pb-[60px]">
            <Footer />
          </div>
        </div>
        <button
          className={`text-errorFG font-bold sticky bottom-[10px] h-[50px] bg-pageBG rounded-full px-4 mt-2
            ${children?.error ? 'visible' : ' invisible'}`}
          disabled={!children.error}
          style={{boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 10px'}}
          onClick={() => setShowError(true)}
        >
          Warning!
        </button>
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
        {showError && (
          <ModalBase bg="bg-red-100" onClose={() => setShowError(false)}>
            {() => <div className="  rounded-lg p-2">{children?.error}</div>}
          </ModalBase>
        )}
      </div>
    )
  }
)
