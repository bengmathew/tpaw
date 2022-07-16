import {Document} from '@contentful/rich-text-types'
import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {Contentful} from '../../Utils/Contentful'
import {
  applyPaddingToHTMLElement,
  applyRectSizingToHTMLElement,
  Padding,
  RectExt,
} from '../../Utils/Geometry'
import {useAssertConst} from '../../Utils/UseAssertConst'
import {fGet} from '../../Utils/Utils'
import {useSimulation} from '../App/WithSimulation'
import {ModalBase} from '../Common/Modal/ModalBase'
import {ParamsInputType} from './ParamsInput/Helpers/ParamsInputType'
import {usePlanContent} from './Plan'

export type GuidePanelStateful = {
  setTransition: (transition: number) => void
}

type Props = {
  layout: 'mobile' | 'desktop' | 'laptop'
  type: ParamsInputType
  sizing: (transition: number) => {
    position: RectExt
    padding: Padding
    headingMarginBottom: number
  }
  transitionRef: React.MutableRefObject<{
    transition: number
  }>
}

export const GuidePanel = React.memo(
  React.forwardRef<GuidePanelStateful, Props>((props, ref) =>
    props.layout === 'mobile' ? (
      <_Mobile {...props} ref={ref} />
    ) : (
      <_LaptopAndDesktop {...props} ref={ref} />
    )
  )
)

const _Mobile = React.memo(
  React.forwardRef<GuidePanelStateful, Props>(
    ({type, sizing, transitionRef}: Props, forwardRef) => {
      const {params} = useSimulation()
      const [show, setShow] = useState(false)
      const outerRef = useRef<HTMLButtonElement | null>(null)
      const [inner, setInner] = useState<HTMLDivElement | null>(null)
      const setTransition = useCallback(
        (transition: number) => {
          const {position, padding, headingMarginBottom} = sizing(transition)
          // Outer.
          applyRectSizingToHTMLElement(position, fGet(outerRef.current))
          fGet(outerRef.current).style.display =
            transition === 0 ? 'none' : 'block'

          // Inner.
          if (inner) {
            applyPaddingToHTMLElement(sizing(1).padding, inner)
          }
        },
        [sizing, inner]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const content = usePlanContent()[type]
      return (
        <>
          <button
            className="absolute rounded-full flex item-center justify-center bg-pageBG"
            ref={outerRef}
            style={{boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 10px'}}
            onClick={() => setShow(true)}
          >
            <h2 className="font-bold">Guide</h2>
          </button>
          {show && (
            <ModalBase onClose={() => setShow(false)}>
              {transitionOut => (
                <div className="" ref={setInner}>
                  <_RichText className="">
                    {content.body[params.strategy]}
                  </_RichText>
                </div>
              )}
            </ModalBase>
          )}
        </>
      )
    }
  )
)

const _LaptopAndDesktop = React.memo(
  React.forwardRef<GuidePanelStateful, Props>(
    ({type, sizing, transitionRef}: Props, forwardRef) => {

      const {params} = useSimulation()
      const outerRef = useRef<HTMLDivElement | null>(null)
      // const innerRef = useRef<HTMLDivElement | null>(null)
      const headerRef = useRef<HTMLHeadingElement | null>(null)
      const setTransition = useCallback(
        (transition: number) => {
          const {position, padding, headingMarginBottom} = sizing(transition)
          // Outer.
          applyRectSizingToHTMLElement(position, fGet(outerRef.current))
          applyPaddingToHTMLElement(padding, fGet(outerRef.current))
          fGet(outerRef.current).style.opacity = `${transition}`
          fGet(outerRef.current).style.display =
            transition === 0 ? 'none' : 'block'

          // Heading.
          fGet(
            headerRef.current
          ).style.marginBottom = `${headingMarginBottom}px`
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const content = usePlanContent()[type]
      return (
        <div className="absolute overflow-scroll" ref={outerRef}>
          <h2 className="uppercase font-bold " ref={headerRef}>
            Guide
          </h2>
          {content.body && (
            <div className={` `}>
              <_RichText className="">
                {content.body[params.strategy]}
              </_RichText>
            </div>
          )}
        </div>
      )
    }
  )
)

const _RichText = React.memo(
  ({className = '', children}: {className?: string; children: Document}) => {
    return (
      <div className={`${className}`}>
        <Contentful.RichText
          body={children}
          ul="list-disc ml-5"
          ol="list-decimal ml-5"
          p="p-base mb-3"
          h1="font-bold text-lg mb-3"
          h2="font-bold text-lg mt-6 mb-3"
          a="underline"
          aExternalLink="text-[12px] ml-1"
        />
      </div>
    )
  }
)
