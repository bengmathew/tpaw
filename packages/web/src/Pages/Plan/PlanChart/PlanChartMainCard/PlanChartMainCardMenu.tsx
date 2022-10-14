import {faChevronRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import Link from 'next/link'
import React, {
  CSSProperties,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import {Contentful} from '../../../../Utils/Contentful'
import {
  applyOriginToHTMLElement,
  Padding,
  rectExt,
  RectExt,
} from '../../../../Utils/Geometry'
import {assert, fGet, noCase} from '../../../../Utils/Utils'
import {useChartData} from '../../../App/WithChartData'
import {useSimulation} from '../../../App/WithSimulation'
import {useWindowSize} from '../../../App/WithWindowSize'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartReact, ChartReactStatefull} from '../../../Common/Chart/ChartReact'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {usePlanContent} from '../../Plan'
import { PlanChartInternalTransitionState } from '../PlanChart'
import {
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
  PlanChartType,
} from '../PlanChartType'
import {TPAWChartDataMain} from '../TPAWChart/TPAWChartDataMain'
import {useGetPlanChartURL} from '../UseGetPlanChartURL'
import {usePlanChartType} from '../UsePlanChartType'
import {planChartLabel} from './PlanChartLabel'
import {PlanChartMainCardMenuButton} from './PlanChartMainCardMenuButton'

const duration = 500
const scale = 0.95
const widthForFullSize = 700
export type PlanChartMainCardMenuStateful = {
  setButtonScale: (scale: number) => void
}
export const PlanChartMainCardMenu = React.memo(
  ({
    className = '',
    style,
    layout,
    transition,
  }: {
    className?: string
    style?: CSSProperties
    layout: 'mobile' | 'desktop' | 'laptop'
    transition: {target: PlanChartInternalTransitionState; duration: number}
  }) => {
    const windowSize = useWindowSize()
    const width = Math.min(windowSize.width, widthForFullSize)
    const simulation = useSimulation()
    const {params} = simulation.tpawResult.args
    const type = usePlanChartType()

    const [referenceElement, setReferenceElement] =
      useState<HTMLDivElement | null>(null)

    const [popperElement, setPopperElement] = useState<HTMLDivElement | null>(
      null
    )
    // Hack to force redraw on open. Seemed like the draws were not taking
    // effect when the canvas was not visible.
    const [drawKey, setDrawKey] = useState(0)
    const [show, setShow] = useState(false)

    const handleShow = () => {
      setDrawKey(x => x + 1)
      setShow(true)
      const position = fGet(referenceElement).getBoundingClientRect()
      const origin = {
        y: position.top,
        x:
          width < widthForFullSize
            ? 0
            : Math.min(position.left, windowSize.width - width - 20),
      }
      applyOriginToHTMLElement(origin, fGet(popperElement))
    }

    const buttonProps = {
      currType: type,
      onHide: () => setShow(false),
      drawKey,
    }

    const [opacity0AtTransitionEnd, setOpacity0AtTransitionEnd] = useState(true)
    const invisible = !show && opacity0AtTransitionEnd

    return (
      <>
        <PlanChartMainCardMenuButton
          ref={setReferenceElement}
          className={className}
          style={style}
          onClick={handleShow}
          layout={layout}
          transition={transition}
        />

        {ReactDOM.createPortal(
          <div
            className=" page fixed inset-0"
            style={{
              visibility: invisible ? 'hidden' : 'visible',
              transitionProperty: 'opacity',
              transitionDuration: `${duration}ms`,
              opacity: show ? '1' : '0',
            }}
            onTransitionEnd={() => setOpacity0AtTransitionEnd(!show)}
          >
            <div
              className="fixed inset-0 bg-black opacity-70"
              onClick={() => setShow(false)}
            />
            <div
              className={`flex absolute flex-col  rounded-xl  bg-pageBG  max-h-[calc(100vh-150px)] overflow-scroll`}
              ref={setPopperElement}
              style={{
                transitionProperty: 'transform',
                transitionDuration: `${duration}ms`,
                transform: `scale(${show ? 1 : scale})`,
                width: `${width}px`,
                boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
              }}
            >
              <_Link type="spending-total" {...buttonProps} />
              {(params.original.extraSpending.essential.length > 0 ||
                params.original.extraSpending.discretionary.length > 0) && (
                <div className=" flex flex-col border- border-gray-400 ">
                  <_Link
                    className="pl-10"
                    type="spending-general"
                    {...buttonProps}
                  />
                  {params.original.extraSpending.essential.map(x => (
                    <_Link
                      className="pl-10"
                      key={`essential-${x.id}`}
                      type={`spending-essential-${x.id}`}
                      {...buttonProps}
                    />
                  ))}
                  {params.original.extraSpending.discretionary.map(x => (
                    <_Link
                      className="pl-10"
                      key={`discretionary-${x.id}`}
                      type={`spending-discretionary-${x.id}`}
                      {...buttonProps}
                    />
                  ))}
                </div>
              )}
              <_Link type={'portfolio'} {...buttonProps} />
              <_Link
                type={'asset-allocation-savings-portfolio'}
                {...buttonProps}
              />
              {!params.original.risk.useTPAWPreset &&
                params.strategy === 'TPAW' && (
                  <_Link
                    type={'asset-allocation-total-portfolio'}
                    {...buttonProps}
                  />
                )}
              <_Link type={'withdrawal'} {...buttonProps} />
            </div>
          </div>,
          window.document.body
        )}
      </>
    )
  }
)

const _Link = React.memo(
  ({
    className = '',
    currType,
    onHide,
    type,
    drawKey,
  }: {
    className?: string
    currType: PlanChartType
    onHide: () => void
    type: Exclude<PlanChartType, 'reward-risk-ratio-comparison'>
    drawKey: number
  }) => {
    const {tpawResult} = useSimulation()
    const {params} = tpawResult.args
    const getPlanChartURL = useGetPlanChartURL()
    const chartData = fGet(useChartData().byYearsFromNowPercentiles.get(type))
    const [description] = useInfo(type)
    const {label, subLabel} = planChartLabel(params.original, type, 'short')
    const isCurrent = _.isEqual(currType, type)
    const windowSize = useWindowSize()
    const width = windowSize.width < 640 ? 120 : 145
    const height = windowSize.width < 640 ? 50 : 55
    return (
      <Link href={getPlanChartURL(type)} shallow>
        <a
          className={`${className} text-left px-4 py-2  grid  gap-x-4 items-center
          ${isCurrent ? 'bg-gray-200 ' : ''} `}
          onClick={() => onHide()}
          style={{grid: 'auto / 1fr auto'}}
        >
          <div className="">
            <h2 className="text-base sm:text-lg font-bold">
              {label.map((x, i) => (
                <React.Fragment key={i}>
                  <span>{x}</span>
                  {i !== label.length - 1 && (
                    <FontAwesomeIcon
                      className="mx-2 text-xs sm:text-sm lighten-2"
                      icon={faChevronRight}
                    />
                  )}
                </React.Fragment>
              ))}
            </h2>
            {subLabel && (
              <h2 className="text-sm sm:text-base font-semibold -mt-1">
                {subLabel}
              </h2>
            )}
            <Contentful.RichText
              body={description[params.strategy]}
              p={`${isCurrent ? '' : 'lighten-2'} text-sm -mt-1`}
            />
          </div>
          <div
            className={`relative  border rounded-xl  flex flex-col justify-center
            ${
              isCurrent
                ? 'bg-gray-300 border-gray-300 '
                : 'bg-gray-200 border-gray-300 '
            }`}
            style={{width: `${width}px`, height: `${height}px`}}
          >
            <_Chart
              data={chartData}
              isCurrent={isCurrent}
              drawKey={drawKey}
              startingSizing={{
                position: rectExt({x: 0, y: 0, width, height}),
                padding: {left: 10, right: 10, top: 5, bottom: 5},
              }}
            />
          </div>
        </a>
      </Link>
    )
  }
)

const useInfo = (
  panelType: Exclude<PlanChartType, 'reward-risk-ratio-comparison'>
) => {
  const content = usePlanContent().chart
  switch (panelType) {
    case 'spending-total':
      return [content.spending.total.menu] as const
    case 'spending-general': {
      return [content.spending.regular.menu] as const
    }
    case 'portfolio':
    case 'asset-allocation-savings-portfolio':
    case 'asset-allocation-total-portfolio':
      return [content[panelType].menu] as const
    case 'withdrawal':
      return [content.withdrawalRate.menu] as const
    default:
      if (isPlanChartSpendingEssentialType(panelType)) {
        return [content.spending.essential.menu] as const
      }
      if (isPlanChartSpendingDiscretionaryType(panelType)) {
        return [content.spending.discretionary.menu] as const
      }
      noCase(panelType)
  }
}

const _Chart = React.memo(
  ({
    data,
    isCurrent,
    drawKey,
    startingSizing,
  }: {
    data: TPAWChartDataMain
    isCurrent: boolean
    drawKey: number
    startingSizing: {position: RectExt; padding: Padding}
  }) => {
    const ref = useRef<ChartReactStatefull<TPAWChartDataMain>>(null)
    useEffect(() => {
      if (!ref.current) return
      ref.current.setState(
        data,
        {
          x: data.years.displayRange,
          y: data.yDisplayRange,
        },
        null
      )
    }, [data, drawKey])

    const components = useCallback(
      () => [
        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 0.5 * 0.8,
          strokeStyle: isCurrent
            ? ChartUtils.color.gray[400]
            : ChartUtils.color.gray[400],
          dataFn: (data: TPAWChartDataMain) => {
            assert(data.series.type === 'percentiles')
            return {
              lines: data.series.percentiles
                .filter(x => !x.isHighlighted)
                .map(x => x.data),
            }
          },
        }),

        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 1.2 * 0.8,
          strokeStyle: isCurrent
            ? ChartUtils.color.gray[500]
            : ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) => {
            assert(data.series.type === 'percentiles')
            return {
              lines: data.series.percentiles
                .filter(x => x.isHighlighted)
                .map(x => x.data),
            }
          },
        }),
      ],
      [isCurrent]
    )

    return (
      <ChartReact<TPAWChartDataMain>
        ref={ref}
        starting={{
          data,
          xyRange: {
            x: data.years.displayRange,
            y: data.yDisplayRange,
          },
          sizing: startingSizing,
        }}
        components={components}
      />
    )
  }
)