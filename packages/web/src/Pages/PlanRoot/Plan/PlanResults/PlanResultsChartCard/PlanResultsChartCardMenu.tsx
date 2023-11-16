import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import _ from 'lodash'
import Link from 'next/link'
import React, { CSSProperties, useEffect, useRef, useState } from 'react'
import ReactDOM from 'react-dom'
import {
  Padding,
  RectExt,
  applyOriginToHTMLElement,
  rectExt,
} from '../../../../../Utils/Geometry'
import { fGet } from '../../../../../Utils/Utils'
import { useWindowSize } from '../../../../App/WithWindowSize'
import { chartDrawDataLines } from '../../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { chartDrawDataRangeBand } from '../../../../Common/Chart/ChartComponent/ChartDrawRangeBand'
import {
  ChartReact,
  ChartReactStatefull,
} from '../../../../Common/Chart/ChartReact'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanColors, usePlanColors } from '../../UsePlanColors'
import { useChartData } from '../../WithChartData'
import { PlanResultsTransitionState } from '../PlanResults'
import { PlanResultsChartType } from '../PlanResultsChartType'
import { TPAWChartDataMain } from '../TPAWChart/TPAWChartDataMain'
import { useGetPlanResultsChartURL } from '../UseGetPlanResultsChartURL'
import { usePlanResultsChartType } from '../UsePlanResultsChartType'
import { planResultsChartLabel } from './PlanResultsChartLabel'
import { PlanResultsChartCardMenuButton } from './PlanResutlsChartCardMenuButton'

const duration = 500
const scale = 0.95
const maxWidth = 700
export type PlanChartMainCardMenuStateful = {
  setButtonScale: (scale: number) => void
}
export const PlanResultsChartCardMenu = React.memo(
  ({
    className = '',
    style,
    transition,
  }: {
    className?: string
    style?: CSSProperties
    transition: { target: PlanResultsTransitionState; duration: number }
  }) => {
    const windowSize = useWindowSize()
    const width = Math.min(windowSize.width, maxWidth)
    const simulation = useSimulation()
    const { params } = simulation.tpawResult
    const type = usePlanResultsChartType()
    const planColors = usePlanColors()

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
        <PlanResultsChartCardMenuButton
          ref={referenceElementRef}
          className={className}
          style={style}
          onClick={handleShow}
          transition={transition}
        />

        {ReactDOM.createPortal(
          <div
            className={clsx(
              ' page fixed inset-0 ',
              // Not setting 'pointer-events-none' was causing an issue on Safari
              // where elements were not scrollable under this even thought it was
              // hidden.
              invisible && 'pointer-events-none',
            )}
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
                backgroundColor: planColors.results.menu.bg,
                color: planColors.results.fg,
              }}
            >
              <_Link type="spending-total" {...buttonProps} />
              {(_.values(
                params.original.adjustmentsToSpending.extraSpending.essential,
              ).length > 0 ||
                _.values(
                  params.original.adjustmentsToSpending.extraSpending
                    .discretionary,
                ).length > 0) && (
                <div className=" flex flex-col  ">
                  <_Link
                    className="pl-10"
                    type="spending-general"
                    {...buttonProps}
                  />
                  {_.values(
                    params.original.adjustmentsToSpending.extraSpending
                      .essential,
                  )
                    .sort((a, b) => a.sortIndex - b.sortIndex)
                    .map((x) => (
                      <_Link
                        className="pl-10"
                        key={`essential-${x.id}`}
                        type={`spending-essential-${x.id}`}
                        {...buttonProps}
                      />
                    ))}
                  {_.values(
                    params.original.adjustmentsToSpending.extraSpending
                      .discretionary,
                  )
                    .sort((a, b) => a.sortIndex - b.sortIndex)
                    .map((x) => (
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
    currType: PlanResultsChartType
    onHide: () => void
    type: PlanResultsChartType
    drawKey: number
  }) => {
    const { tpawResult } = useSimulation()
    const { params } = tpawResult
    const getPlanChartURL = useGetPlanResultsChartURL()
    const chartData = fGet(useChartData().byYearsFromNowPercentiles.get(type))

    const { label, subLabel, description } = planResultsChartLabel(
      params.original,
      type,
      'short',
    )
    const isCurrent = _.isEqual(currType, type)
    const windowSize = useWindowSize()
    const width = windowSize.width < 640 ? 120 : 145
    const height = windowSize.width < 640 ? 50 : 55
    const planColors = usePlanColors()
    return (
      <Link
        className={clsx(
          className,
          'text-left px-4 py-2  grid  gap-x-4 items-center',
        )}
        onClick={() => onHide()}
        style={{
          grid: 'auto / 1fr auto',
          backgroundColor: isCurrent
            ? planColors.results.menu.currentBG
            : undefined,
        }}
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
          className={`relative  border rounded-xl  flex flex-col justify-center `}
          style={{
            width: `${width}px`,
            height: `${height}px`,
            backgroundColor: isCurrent
              ? planColors.results.menu.chart.bg
              : planColors.results.menu.chart.bg,
          }}
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
    const planColors = usePlanColors()

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
        components={components(planColors)}
      />
    )
  },
)

const components =
  ({ results: chartColors }: PlanColors) =>
  () => [
    chartDrawDataRangeBand<TPAWChartDataMain>({
      fillStyle: chartColors.rangeBand,

      dataFn: (data: TPAWChartDataMain) => ({
        min: fGet(_.first(data.percentiles)).data,
        max: fGet(_.last(data.percentiles)).data,
      }),
    }),

    chartDrawDataLines<TPAWChartDataMain>({
      lineWidth: 1.2 * 0.8,
      strokeStyle: chartColors.medianLine,
      dataFn: (data: TPAWChartDataMain) => {
        return {
          lines: data.percentiles
            .filter((x) => x.percentile === 50)
            .map((x) => x.data),
        }
      },
    }),
  ]
