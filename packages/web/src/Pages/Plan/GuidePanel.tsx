import {Document} from '@contentful/rich-text-types'
import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {Contentful} from '../../Utils/Contentful'
import {formatPercentage} from '../../Utils/FormatPercentage'
import {
  applyOriginToHTMLElement,
  Origin,
  Padding,
  paddingCSSStyle,
  Size,
  sizeCSSStyle,
} from '../../Utils/Geometry'
import {useAssertConst} from '../../Utils/UseAssertConst'
import {fGet} from '../../Utils/Utils'
import {useMarketData} from '../App/WithMarketData'
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
  sizing: {
    dynamic: (transition: number) => {
      origin: Origin
    }
    fixed: {
      size: Size
      padding: Padding
      headingMarginBottom: number
    }
  }
  transitionRef: React.MutableRefObject<{
    transition: number
  }>
}
export type GuidePanelSizing = Props['sizing']

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
      const setTransition = useCallback(
        (transition: number) => {
          const {origin} = sizing.dynamic(transition)
          // Outer.
          applyOriginToHTMLElement(origin, fGet(outerRef.current))
          fGet(outerRef.current).style.display =
            transition === 0 ? 'none' : 'block'
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const content = useContent(type)
      const {size, padding} = sizing.fixed
      return (
        <>
          <button
            className="absolute rounded-full flex item-center justify-center bg-pageBG"
            ref={outerRef}
            style={{
              boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 10px',
              ...sizeCSSStyle(size),
            }}
            onClick={() => setShow(true)}
          >
            <h2 className="font-bold">Guide</h2>
          </button>
          {show && (
            <ModalBase onClose={() => setShow(false)}>
              {transitionOut => (
                <div className="" style={{...paddingCSSStyle(padding)}}>
                  <_RichText className="">{content}</_RichText>
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
      const setTransition = useCallback(
        (transition: number) => {
          const {origin} = sizing.dynamic(transition)
          // Outer.
          applyOriginToHTMLElement(origin, fGet(outerRef.current))
          fGet(outerRef.current).style.opacity = `${transition}`
          fGet(outerRef.current).style.display =
            transition === 0 ? 'none' : 'block'
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const content = useContent(type)

      const {padding, headingMarginBottom, size} = sizing.fixed
      return (
        <div
          className="absolute overflow-scroll"
          ref={outerRef}
          style={{
            ...sizeCSSStyle(size),
            ...paddingCSSStyle(padding),
          }}
        >
          <h2
            className="uppercase font-bold "
            style={{marginBottom: `${headingMarginBottom}px`}}
          >
            Guide
          </h2>
          <div className={` `}>
            <_RichText className="">{content}</_RichText>
          </div>
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

function useContent(type: ParamsInputType) {
  const {params, numRuns} = useSimulation()
  const {CAPE, bondRates, inflation} = useMarketData()

  const formatDate = (epoch: number) =>
    new Date(epoch).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })

  const content = usePlanContent()[type].body[params.strategy]
  const variables = {
    numRuns: `${numRuns}`,
    capeDate:formatDate(CAPE.date),
    expectedReturnsStocksCAPE: (CAPE.value.toFixed(2)),
    expectedReturnsStocksOneOverCAPE: formatPercentage(1)(CAPE.oneOverCAPE),
    expectedReturnsStocksRegressionFull5Year: formatPercentage(1)(
      CAPE.regression.full.fiveYear
    ),
    expectedReturnsStocksRegressionFull10Year: formatPercentage(1)(
      CAPE.regression.full.tenYear
    ),
    expectedReturnsStocksRegressionFull20Year: formatPercentage(1)(
      CAPE.regression.full.twentyYear
    ),
    expectedReturnsStocksRegressionFull30Year: formatPercentage(1)(
      CAPE.regression.full.thirtyYear
    ),
    expectedReturnsStocksRegressionRestricted5Year: formatPercentage(1)(
      CAPE.regression.restricted.fiveYear
    ),
    expectedReturnsStocksRegressionRestricted10Year: formatPercentage(1)(
      CAPE.regression.restricted.tenYear
    ),
    expectedReturnsStocksRegressionRestricted20Year: formatPercentage(1)(
      CAPE.regression.restricted.twentyYear
    ),
    expectedReturnsStocksRegressionRestricted30Year: formatPercentage(1)(
      CAPE.regression.restricted.thirtyYear
    ),
    expectedReturnsRegresssionAverage: formatPercentage(1)(CAPE.regressionAverage),
    expectedReturnsSuggested: formatPercentage(1)(CAPE.suggested),
    bondsDate:formatDate(bondRates.date),
    expectedReturnsBonds5Year: formatPercentage(1)(bondRates.fiveYear),
    expectedReturnsBonds10Year: formatPercentage(1)(bondRates.tenYear),
    expectedReturnsBonds20Year: formatPercentage(1)(bondRates.twentyYear),
    expectedReturnsBonds30Year: formatPercentage(1)(bondRates.thirtyYear),
    inflationDate: formatDate(inflation.date),
    inflation: formatPercentage(1)(inflation.value),
  }
  return Contentful.replaceVariables(variables, content)
}
