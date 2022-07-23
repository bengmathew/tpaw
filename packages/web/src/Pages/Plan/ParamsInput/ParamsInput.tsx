import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyOriginToHTMLElement,
  Origin,
  Padding,
  Size,
  sizeCSSStyle,
} from '../../../Utils/Geometry'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet, noCase} from '../../../Utils/Utils'
import {ChartPanelType} from '../ChartPanel/ChartPanelType'
import {ParamsInputType} from './Helpers/ParamsInputType'
import {ParamsInputAgeAndRetirement} from './ParamsInputAgeAndRetirement/ParamsInputAgeAndRetirement'
import {ParamsInputBodyPassThruProps} from './ParamsInputBody'
import {ParamsInputCompareStrategies} from './ParamsInputCompareStrategies'
import {ParamsInputCurrentPortfolioBalance} from './ParamsInputCurrentPortfolioBalance'
import {ParamsInputDev} from './ParamsInputDev'
import {ParamsInputExpectedReturns} from './ParamsInputExpectedReturns'
import {ParamsInputExtraSpending} from './ParamsInputExtraSpending'
import {ParamsInputFutureSavings} from './ParamsInputFutureSavings'
import {ParamsInputIncomeDuringRetirement} from './ParamsInputIncomeDuringRetirement'
import {ParamsInputInflation} from './ParamsInputInflation'
import {ParamsInputLegacy} from './ParamsInputLegacy'
import {ParamsInputLMP} from './ParamsInputLMP'
import {ParamsInputSimulation} from './ParamsInputSimulation'
import {ParamsInputSpendingCeilingAndFloor} from './ParamsInputSpendingCeilingAndFloor'
import {ParamsInputSpendingTilt} from './ParamsInputSpendingTilt'
import {ParamsInputStockAllocation} from './ParamsInputStockAllocation'
import {ParamsInputWithdrawalRate} from './ParamsInputWithdrawal'

type Props = {
  layout: 'laptop' | 'desktop' | 'mobile'
  sizing: {
    dynamic: (transition: number) => {
      origin: Origin
    }
    fixed: {
      size: Size
      padding: Padding
      cardPadding: Padding
      headingMarginBottom: number
    }
  }
  transitionRef: React.MutableRefObject<{
    transition: number
  }>
  paramInputType: ParamsInputType
  setState: (state: 'summary' | ParamsInputType) => void
  chartType: ChartPanelType | 'sharpe-ratio'
  setChartType: (type: ChartPanelType | 'sharpe-ratio') => void
}
export type ParamsInputSizing = Props['sizing']

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
        setState,
        chartType,
        setChartType,
        transitionRef,
      }: Props,
      forwardRef
    ) => {
      const outerRef = useRef<HTMLDivElement | null>(null)

      const [transitionCount, setTransitionCount] = useState(0)
      const setTransition = useCallback(
        (transition: number) => {
          const {origin} = sizing.dynamic(transition)
          const outer = fGet(outerRef.current)
          applyOriginToHTMLElement(origin, outer)
          outer.style.opacity = `${transition}`
          outer.style.display = transition === 0 ? 'none' : 'grid'
          if (transition === 0) setTransitionCount(x => x + 1)
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const {size, padding, cardPadding, headingMarginBottom} = sizing.fixed

      return (
        <div
          className={`absolute`}
          ref={outerRef}
          style={{...sizeCSSStyle(size)}}
        >
          <_Body
            key={transitionCount}
            sizing={{padding, cardPadding, headingMarginBottom}}
            layout={layout}
            type={paramInputType}
            onDone={() => setState('summary')}
            chartType={chartType}
            setChartType={setChartType}
          />
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
    chartType: ChartPanelType | 'sharpe-ratio'
    setChartType: (type: ChartPanelType | 'sharpe-ratio') => void
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
      case 'legacy':
        return <ParamsInputLegacy {...props} />
      case 'stock-allocation':
        return <ParamsInputStockAllocation {...props} />
      case 'spending-tilt':
        return <ParamsInputSpendingTilt {...props} />
      case 'spending-ceiling-and-floor':
        return <ParamsInputSpendingCeilingAndFloor {...props} />
      case 'lmp':
        return <ParamsInputLMP {...props} />
      case 'withdrawal':
        return <ParamsInputWithdrawalRate {...props} />
      case 'compare-strategies':
        return (
          <ParamsInputCompareStrategies
            chartType={chartType}
            setChartType={setChartType}
            {...props}
          />
        )
      case 'expected-returns':
        return <ParamsInputExpectedReturns {...props} />
      case 'inflation':
        return <ParamsInputInflation {...props} />
      case 'simulation':
        return <ParamsInputSimulation {...props} />
      case 'dev':
        return <ParamsInputDev {...props} />
      default:
        noCase(type)
    }
  }
)
