import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyPaddingToHTMLElement,
  applyRectSizingToHTMLElement,
  Padding,
  RectExt,
} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet, noCase} from '../../../Utils/Utils'
import {ChartPanelType} from '../ChartPanel/ChartPanelType'
import {ParamsInputType} from './Helpers/ParamsInputType'
import {ParamsInputAgeAndRetirement} from './ParamsInputAgeAndRetirement/ParamsInputAgeAndRetirement'
import {ParamsInputBodyPassThruProps} from './ParamsInputBody'
import {ParamsInputCurrentPortfolioBalance} from './ParamsInputCurrentPortfolioBalance'
import { ParamsInputDev } from './ParamsInputDev'
import {ParamsInputExpectedReturns} from './ParamsInputExpectedReturns'
import {ParamsInputExtraSpending} from './ParamsInputExtraSpending'
import {ParamsInputFutureSavings} from './ParamsInputFutureSavings'
import {ParamsInputIncomeDuringRetirement} from './ParamsInputIncomeDuringRetirement'
import {ParamsInputInflation} from './ParamsInputInflation'
import {ParamsInputLegacy} from './ParamsInputLegacy'
import {ParamsInputRiskAndTimePreference} from './ParamsInputRiskAndTimePreference'
import {ParamsInputSpendingCeilingAndFloor} from './ParamsInputSpendingCeilingAndFloor'
import {ParamsInputStrategy} from './ParamsInputStrategy'
import {ParamsInputSummary} from './ParamsInputSummary'
import {Reset} from './Reset'
import {Share} from './Share'

type Props = {
  layout: 'laptop' | 'desktop' | 'mobile'
  sizing: (transition: number) => {
    position: RectExt
    padding: Padding
    cardPadding: Padding
    headingMarginBottom: number
  }
  transitionRef: React.MutableRefObject<{
    transition: number
  }>
  state: ParamsInputType | 'summary'
  paramInputType: ParamsInputType
  setState: (state: 'summary' | ParamsInputType) => void
  chartType: ChartPanelType
  setChartType: (type: ChartPanelType) => void
}

export type ParamsInputStateful = {
  setTransition: (transition: number) => void
}
export const ParamsInput = React.memo(
  React.forwardRef<ParamsInputStateful, Props>(
    (
      {
        layout,
        sizing,
        paramInputType,
        state,
        setState,
        chartType,
        setChartType,
        transitionRef,
      }: Props,
      forwardRef
    ) => {
      const outerRef = useRef<HTMLDivElement | null>(null)
      const summaryRef = useRef<HTMLDivElement | null>(null)
      const detailRef = useRef<HTMLDivElement | null>(null)

      const [transitionCount, setTransitionCount] = useState(0)
      const setTransition = useCallback(
        (transition: number) => {
          const {position} = sizing(transition)
          const summary = fGet(summaryRef.current)
          const detail = fGet(detailRef.current)
          const at0 = sizing(0)
          const at1 = sizing(1)
          applyRectSizingToHTMLElement(position, fGet(outerRef.current))

          // Summary
          summary.style.opacity = `${linearFnFomPoints(0, 1, 1, 0)(transition)}`
          summary.style.display = transition === 1 ? 'none' : 'block'
          applyPaddingToHTMLElement(at0.padding, summary)

          // Detail.
          detail.style.opacity = `${transition}`
          detail.style.display = transition === 0 ? 'none' : 'grid'
          if (transition === 0) {
            setTransitionCount(x => x + 1)
          }
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const sizingAt0 = sizing(0)
      const sizingAt1 = sizing(1)

      return (
        <div className="absolute overflow-hidden bg-planBG " ref={outerRef}>
          <div
            className={`absolute top-0 left-0 h-full overflow-y-scroll `}
            style={{width: `${sizingAt0.position.width}px`}}
            ref={summaryRef}
          >
            <div
              className="flex justify-between items-center"
              style={{
                marginBottom: `${sizingAt0.headingMarginBottom}px`,
                paddingLeft: `${sizingAt0.cardPadding.left}px`,
              }}
            >
              <h2
                className={`uppercase font-bold 
                ${layout !== 'laptop' ? 'invisible' : 'invisible'}`}
              >
                Input
              </h2>
              <div className={`flex gap-x-4 `}>
                <Reset />
                <Share />
              </div>
            </div>

            <ParamsInputSummary
              layout={layout}
              state={state}
              setState={setState}
              cardPadding={sizingAt0.cardPadding}
            />
          </div>
          <div
            className={`absolute top-0 left-0 h-full grid`}
            style={{
              width: `${sizingAt1.position.width}px`,
              grid: '1fr/1fr',
            }}
            ref={detailRef}
          >
            <_Body
              key={transitionCount}
              sizing={sizingAt1}
              layout={layout}
              type={paramInputType}
              onDone={() => setState('summary')}
              chartType={chartType}
              setChartType={setChartType}
            />
          </div>
        </div>
      )
    }
  )
)

const _Body = React.memo(
  ({
    sizing,
    type,
    layout,
    onDone,
    chartType,
    setChartType,
  }: {
    sizing: {
      padding: Padding
      cardPadding: Padding
      headingMarginBottom: number
    }
    layout: 'desktop' | 'mobile' | 'laptop'
    type: ParamsInputType
    onDone: () => void
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  }) => {
    const props: ParamsInputBodyPassThruProps = {layout, sizing}

    switch (type) {
      case 'age-and-retirement':
        return <ParamsInputAgeAndRetirement {...props} />
      case 'current-portfolio-balance':
        return <ParamsInputCurrentPortfolioBalance {...props} />
      case 'future-savings':
        return <ParamsInputFutureSavings onBack={onDone} {...props} />
      case 'income-during-retirement':
        return <ParamsInputIncomeDuringRetirement {...props} />
      case 'extra-spending':
        return (
          <ParamsInputExtraSpending
            chartType={chartType}
            setChartType={setChartType}
            {...props}
          />
        )
      case 'spending-ceiling-and-floor':
        return <ParamsInputSpendingCeilingAndFloor {...props} />
      case 'legacy':
        return <ParamsInputLegacy {...props} />
      case 'risk-and-time-preference':
        return <ParamsInputRiskAndTimePreference {...props} />
      case 'strategy':
        return <ParamsInputStrategy {...props} />
      case 'expected-returns':
        return <ParamsInputExpectedReturns {...props} />
      case 'inflation':
        return <ParamsInputInflation {...props} />
      case 'dev':
        return <ParamsInputDev {...props} />
      default:
        noCase(type)
    }
  }
)
