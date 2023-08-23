import { Transition } from '@headlessui/react'
import React, {
  MouseEvent,
  ReactElement,
  ReactNode,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  newPaddingHorz,
  paddingCSS,
  paddingCSSStyleHorz,
} from '../../../../../Utils/Geometry'
import { assert, fGet } from '../../../../../Utils/Utils'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../Helpers/PlanInputType'
import { PlanInputSizing } from '../PlanInput'
import { PlanInputBodyGuide } from './PlanInputBodyGuide/PlanInputBodyGuide'
import { usePlanInputGuideContent } from './PlanInputBodyGuide/UsePlanInputGuideContent'
import { PlanInputBodyHeader } from './PlanInputBodyHeader'

const duration = 500

export const PlanInputLaptopAndDesktop = React.memo(
  ({
    layout,
    sizing,
    children,
    type,
    onBackgroundClick,
    customGuideIntro,
  }: {
    layout: 'laptop' | 'desktop'
    sizing: PlanInputSizing['fixed']
    type: PlanInputType
    customGuideIntro?: ReactNode
    onBackgroundClick?: () => void
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const [state, setState] = useState<
      | { mainOrInput: 'main'; onDone: (() => void) | null }
      | { mainOrInput: 'input' }
    >({ mainOrInput: 'main', onDone: null })
    const { planParams } = useSimulation()
    const hasInput = children?.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) {
        setState({ mainOrInput: 'input' })
      }
    }, [hasInput])

    const mainScrollRef = useRef<HTMLDivElement>(null)
    const backgroundDivRef = useRef<HTMLDivElement>(null)
    const guideContent = usePlanInputGuideContent(type)
    const inputScrollRef = useRef<HTMLDivElement>(null)

    const { padding } =
      planParams.dialogPosition !== 'done'
        ? sizing.dialogMode
        : sizing.notDialogMode
    const { cardPadding } = sizing
    return (
      <>
        {/*  Main container. Main and input needs separate scroll containers, 
        so they don't interfere with each other's scroll. */}
        <Transition
          ref={mainScrollRef}
          show={state.mainOrInput === 'main'}
          className="absolute inset-0 overflow-y-scroll"
          enterFrom="opacity-0  translate-x-[-15px]"
          enterTo="opacity-100  translate-x-0"
          leaveFrom="opacity-100 translate-x-0"
          leaveTo="opacity-0 translate-x-[-15px]"
          style={{
            transitionProperty: 'opacity, transform',
            transitionDuration: `${duration}ms`,
          }}
          onClick={(e: MouseEvent) => {
            if (e.target === mainScrollRef?.current) onBackgroundClick?.()
          }}
        >
          <div
            ref={backgroundDivRef}
            className="mb-20 "
            // Padding should be inside scroll container to place scrollbar at
            // edge.
            style={{
              ...paddingCSSStyleHorz(newPaddingHorz(padding)),
              paddingTop: `${padding.top}px`,
            }}
            onClick={(e) => {
              if (e.target === backgroundDivRef.current) onBackgroundClick?.()
            }}
          >
            <PlanInputBodyHeader
              className="mb-6 z-10"
              type={type}
              onBackgroundClick={onBackgroundClick}
            />
            {guideContent && (
              <PlanInputBodyGuide
                className="mb-10"
                type={type}
                padding={cardPadding}
                customIntro={customGuideIntro}
              />
            )}
            {children.content}
            {children?.error && (
              <div className=" sticky bottom-0 pt-10">
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
            ref={inputScrollRef}
            show={state.mainOrInput === 'input'}
            className={`absolute inset-0  overflow-y-scroll`}
            enterFrom="opacity-0  translate-x-[15px]"
            enterTo="opacity-100  translate-x-0"
            leaveFrom="opacity-100 translate-x-0"
            leaveTo="opacity-0 translate-x-[15px]"
            style={{
              transitionProperty: 'opacity, transform',
              transitionDuration: `${duration}ms`,
              // Padding should be inside scroll container to place scrollbar at
              // edge.
              ...paddingCSSStyleHorz(newPaddingHorz(padding)),
            }}
            afterLeave={() => {
              assert(state.mainOrInput === 'main')
              if (state.onDone) {
                state.onDone()
                setState({ mainOrInput: 'main', onDone: null })
              }
            }}
            onClick={(e: MouseEvent) => {
              if (e.target === inputScrollRef?.current) onBackgroundClick?.()
            }}
          >
            <div
              className={`my-20 bg-cardBG rounded-2xl border-gray-200 border`}
              style={{ padding: paddingCSS(cardPadding) }}
            >
              <div className="">
                {fGet(children.input)((onDone) =>
                  setState({ mainOrInput: 'main', onDone }),
                )}
              </div>
            </div>
          </Transition>
        )}
      </>
    )
  },
)
