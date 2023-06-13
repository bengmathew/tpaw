import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import Link from 'next/link'
import React, { CSSProperties, useEffect, useRef, useState } from 'react'
import ReactDOM from 'react-dom'
import {
  Padding,
  RectExt,
  applyOriginToHTMLElement,
  rectExt,
} from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { useChartData } from '../../../App/WithChartData'
import { useSimulation } from '../../../App/WithSimulation'
import { useWindowSize } from '../../../App/WithWindowSize'
import { chartDrawDataLines } from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {
  ChartReact,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { PlanChartTransitionState } from '../PlanChart'
import { PlanChartType } from '../PlanChartType'
import { TPAWChartDataMain } from '../TPAWChart/TPAWChartDataMain'
import { useGetPlanChartURL } from '../UseGetPlanChartURL'
import { usePlanChartType } from '../UsePlanChartType'
import { planChartLabel } from './PlanChartLabel'
import { PlanChartMainCardMenuButton } from './PlanChartMainCardMenuButton'

const duration = 500
const scale = 0.95
const maxWidth = 700
export type PlanChartMainCardMenuStateful = {
  setButtonScale: (scale: number) => void
}
export const PlanChartMainCardMenu = React.memo(
  ({
    className = '',
    style,
    transition,
  }: {
    className?: string
    style?: CSSProperties
    transition: { target: PlanChartTransitionState; duration: number }
  }) => {
    const windowSize = useWindowSize()
    const width = Math.min(windowSize.width, maxWidth)
    const simulation = useSimulation()
    const { params } = simulation.tpawResult
    const type = usePlanChartType()

    const referenceElementRef = useRef<HTMLDivElement | null>(null)
    const popperElementRef = useRef<HTMLDivElement | null>(null)
    // Hack to force redraw on open. Seemed like the draws were not taking
    // effect when the canvas was not visible.
    const [drawKey, setDrawKey] = useState(0)
    const [show, setShow] = useState(false)

    const handleShow = () => {
      setDrawKey((x) => x + 1)
      setShow(true)
      const position = fGet(referenceElementRef.current).getBoundingClientRect()
      const origin = {
        y: position.top,
        x:
          width < maxWidth
            ? 0
            : Math.min(position.left, windowSize.width - width - 20),
      }
      applyOriginToHTMLElement(origin, fGet(popperElementRef.current))
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
          ref={referenceElementRef}
          className={className}
          style={style}
          onClick={handleShow}
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
              ref={popperElementRef}
              style={{
                transitionProperty: 'transform',
                transitionDuration: `${duration}ms`,
                transform: `scale(${show ? 1 : scale})`,
                width: `${width}px`,
                boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
              }}
            >
              <_Link type="spending-total" {...buttonProps} />
              {(params.original.plan.adjustmentsToSpending.extraSpending
                .essential.length > 0 ||
                params.original.plan.adjustmentsToSpending.extraSpending
                  .discretionary.length > 0) && (
                <div className=" flex flex-col border- border-gray-400 ">
                  <_Link
                    className="pl-10"
                    type="spending-general"
                    {...buttonProps}
                  />
                  {params.original.plan.adjustmentsToSpending.extraSpending.essential.map(
                    (x) => (
                      <_Link
                        className="pl-10"
                        key={`essential-${x.id}`}
                        type={`spending-essential-${x.id}`}
                        {...buttonProps}
                      />
                    ),
                  )}
                  {params.original.plan.adjustmentsToSpending.extraSpending.discretionary.map(
                    (x) => (
                      <_Link
                        className="pl-10"
                        key={`discretionary-${x.id}`}
                        type={`spending-discretionary-${x.id}`}
                        {...buttonProps}
                      />
                    ),
                  )}
                </div>
              )}
              <_Link type={'portfolio'} {...buttonProps} />
              <_Link
                type={'asset-allocation-savings-portfolio'}
                {...buttonProps}
              />
              <_Link type={'withdrawal'} {...buttonProps} />
            </div>
          </div>,
          window.document.body,
        )}
      </>
    )
  },
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
    const { tpawResult } = useSimulation()
    const { params } = tpawResult
    const getPlanChartURL = useGetPlanChartURL()
    const chartData = fGet(useChartData().byYearsFromNowPercentiles.get(type))
    const { label, subLabel, description } = planChartLabel(
      params.original,
      type,
      'short',
    )
    const isCurrent = _.isEqual(currType, type)
    const windowSize = useWindowSize()
    const width = windowSize.width < 640 ? 120 : 145
    const height = windowSize.width < 640 ? 50 : 55
    return (
      <Link
        className={`${className} text-left px-4 py-2  grid  gap-x-4 items-center
      ${isCurrent ? 'bg-gray-200 ' : ''} `}
        onClick={() => onHide()}
        style={{ grid: 'auto / 1fr auto' }}
        href={getPlanChartURL(type)}
        shallow
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
          <p className={`${isCurrent ? '' : 'lighten-2'} text-sm -mt-1`}>
            {description}
          </p>
        </div>
        <div
          className={`relative  border rounded-xl  flex flex-col justify-center
            ${
              isCurrent
                ? 'bg-gray-300 border-gray-300 '
                : 'bg-gray-200 border-gray-300 '
            }`}
          style={{ width: `${width}px`, height: `${height}px` }}
        >
          <_Chart
            data={chartData}
            isCurrent={isCurrent}
            drawKey={drawKey}
            startingSizing={{
              position: rectExt({ x: 0, y: 0, width, height }),
              padding: { left: 10, right: 10, top: 5, bottom: 5 },
            }}
          />
        </div>
      </Link>
    )
  },
)

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
    startingSizing: { position: RectExt; padding: Padding }
  }) => {
    const ref = useRef<ChartReactStatefull<TPAWChartDataMain>>(null)
    useEffect(() => {
      if (!ref.current) return
      ref.current.setState(
        data,
        {
          x: data.months.displayRange,
          y: data.yDisplayRange,
        },
        null,
      )
    }, [data, drawKey])

    return (
      <ChartReact<TPAWChartDataMain>
        ref={ref}
        starting={{
          data,
          xyRange: {
            x: data.months.displayRange,
            y: data.yDisplayRange,
          },
          sizing: startingSizing,
        }}
        components={components}
      />
    )
  },
)

const components = () => [
  // chartDrawDataRangeBand<TPAWChartDataMain>({
  //   fillStyle: ChartUtils.color.withOpacity(ChartUtils.color.gray[400], 0.5),
  //   dataFn: (data: TPAWChartDataMain) => ({
  //     min: fGet(_.first(data.percentiles)).data,
  //     max: fGet(_.last(data.percentiles)).data,
  //   }),
  // }),

  chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2 * 0.8,
    strokeStyle: ChartUtils.color.gray[500],
    dataFn: (data: TPAWChartDataMain) => {
      return {
        lines: data.percentiles
          .filter((x) => x.percentile === 50)
          .map((x) => x.data),
      }
    },
  }),
]
