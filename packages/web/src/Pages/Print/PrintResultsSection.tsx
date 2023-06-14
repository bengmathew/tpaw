import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fGet } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { rectExt } from '../../Utils/Geometry'
import { useChartData } from '../App/WithChartData'
import { useSimulation } from '../App/WithSimulation'
import { getYAxisGridInfo } from '../Common/Chart/ChartComponent/ChartYAxis'
import { ChartReactStatefull } from '../Common/Chart/ChartReact'
import { planChartLabel } from '../Plan/PlanChart/PlanChartMainCard/PlanChartLabel'
import {
  planChartLegacyCardFormat,
  usePlanChartLegacyCardData,
} from '../Plan/PlanChart/PlanChartSidePanel/PlanChartSidePanelLegacyCard'
import { PlanChartType } from '../Plan/PlanChart/PlanChartType'
import { TPAWChartDataMain } from '../Plan/PlanChart/TPAWChart/TPAWChartDataMain'
import { tpawChartMainXAxisSizing } from '../Plan/PlanChart/TPAWChart/TPAWChartMain'
import { TPAWChartMainPrint } from '../Plan/PlanChart/TPAWChart/TPAWChartMainPrint'
import { PrintSection } from './PrintSection'

export const PrintResultsSection = React.memo(() => {
  const { params, tpawResult } = useSimulation()
  const { extraSpending } = params.plan.adjustmentsToSpending

  const secondaryCharts: PlanChartType[][] = _.chunk(
    _.compact([
      extraSpending.discretionary.length > 0 ||
      extraSpending.essential.length > 0
        ? 'spending-general'
        : undefined,
      ...extraSpending.essential.map(
        (x) => `spending-essential-${x.id}` as const,
      ),
      ...extraSpending.discretionary.map(
        (x) => `spending-discretionary-${x.id}` as const,
      ),
      'portfolio' as const,
      'asset-allocation-savings-portfolio' as const,
      'withdrawal' as const,
    ]),
    3,
  )
  return (
    <>
      <PrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Your Results</h1>
      </PrintSection>
      <PrintSection>
        <_Chart className="mt-8" type="spending-total" />
        {params.plan.advanced.strategy === 'SWR' && (
          <h2 className=" mt-4 text-right">
            Success Rate:{' '}
            <span className="text-xl">
              {formatPercentage(0)(
                1 - tpawResult.percentageOfRunsWithInsufficientFunds,
              )}
            </span>
          </h2>
        )}
        <_Legacy className="mt-12" />
      </PrintSection>
      {secondaryCharts.map((x, i) => (
        <PrintSection
          key={i}
          className={clsx(
            'flex flex-col',
            x.length === 3 ? 'justify-between' : 'justify-start gap-y-10',
          )}
        >
          {x.map((y, i) => (
            <_Chart key={i} className="" type={y} />
          ))}
        </PrintSection>
      ))}
    </>
  )
})

const _Legacy = React.memo(({ className }: { className?: string }) => {
  const { params } = useSimulation()
  const data = usePlanChartLegacyCardData()
  const maxLegacy = Math.max(...data.map((x) => x.amount))
  const hasLegacy =
    maxLegacy > 0 ||
    params.plan.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0
  return (
    <div className={clsx(className)}>
      <h2 className="text-xl font-bold">Legacy</h2>
      {hasLegacy && (
        <h2 className=" ">These dollars are adjusted for inflation.</h2>
      )}
      <div className="mt-2 inline-block border border-black rounded-lg p-2">
        {hasLegacy ? (
          <>
            <div
              className="inline-grid  mt-0.5 gap-x-4 items-center text-[13px]"
              style={{ grid: 'auto/auto 1fr' }}
            >
              {data.map((x) => (
                <React.Fragment key={x.percentile}>
                  <h2 className="text-right">
                    {x.percentile}
                    <sup className="text-[10px]">th</sup> percentile
                  </h2>
                  <h2 className="font-mono text-right">
                    {planChartLegacyCardFormat(x.amount, 'desktop')}
                  </h2>
                </React.Fragment>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="">
              <span className="text-lg">$0</span>{' '}
              <div className="text-lg">(no legacy target entered)</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
})

const _Chart = React.memo(
  ({ className, type }: { className?: string; type: PlanChartType }) => {
    const { params, tpawResult } = useSimulation()
    const allChartData = useChartData()
    const chartMainData = fGet(allChartData.byYearsFromNowPercentiles.get(type))
    const outerDivRef = useRef<HTMLDivElement>(null)

    const hasPartner = tpawResult.params.people.withPartner
    const { displayMax } = getYAxisGridInfo({
      max: chartMainData.yDisplayRange.end,
      maxGridLine: null,
    })
    const [chart, setChart] =
      useState<ChartReactStatefull<TPAWChartDataMain> | null>(null)
    useEffect(() => {}, [])
    useLayoutEffect(() => {
      if (!chart) return
      const outerDiv = fGet(outerDivRef.current)

      const observer = new ResizeObserver(() => {
        const width = outerDiv.getBoundingClientRect().width
        chart.setSizing(_sizingFromWidth(width, hasPartner), null)
      })
      observer.observe(outerDiv)
      return () => observer.disconnect()
    }, [chart, hasPartner])

    const { label, subLabel, yAxisDescription } = planChartLabel(
      params,
      type,
      'full',
    )
    const yAxisDescriptionStr = yAxisDescription
      ? yAxisDescription.notMobile.map((x) => x.value).join(' ')
      : null

    return (
      <div className={clsx(className, ' break-inside-avoid-page')}>
        <h2 className="flex items-center">
          {label.map((x, i) => (
            <React.Fragment key={i}>
              {i > 0 && (
                <FontAwesomeIcon className="mx-3" icon={faChevronRight} />
              )}
              <span className="text-xl font-bold">{x}</span>
            </React.Fragment>
          ))}
        </h2>
        {subLabel && <h2 className="text-xl font-bold">{subLabel}</h2>}
        <h2 className="">
          {yAxisDescriptionStr && (
            <span className="">{yAxisDescriptionStr}. </span>
          )}
          The graph shows 5<span className=" align-super text-[10px]">th</span>{' '}
          to 95
          <span className=" align-super text-[10px]">th</span> percentiles with
          the 50<span className=" align-super text-[10px]">th</span> percentile
          in bold.
        </h2>

        <div
          className={clsx(
            // The border is applied to the div outside the measured div.
            'relative  border-[1px] border-black rounded-lg mt-2',
          )}
        >
          <div className="h-[225px]" ref={outerDivRef}>
            <TPAWChartMainPrint
              starting={{
                data: chartMainData,
                xyRange: {
                  x: chartMainData.months.displayRange,
                  y: { start: 0, end: displayMax },
                },
                sizing: _sizingFromWidth(300, hasPartner),
              }}
              ref={setChart}
            />
          </div>
        </div>
      </div>
    )
  },
)

const _sizingFromWidth = (width: number, hasPartner: boolean) => {
  const xAxisSizing = tpawChartMainXAxisSizing(width)

  const padding = {
    left: 90,
    right: 10,
    top: 20, // 20 is needed for the max y label.
    bottom:
      2 +
      xAxisSizing.height +
      (hasPartner
        ? 3 * xAxisSizing.gap + xAxisSizing.height
        : 2 * xAxisSizing.gap),
  }

  return {
    position: rectExt({
      x: 0,
      y: 0,
      width,
      height: 225,
    }),
    padding,
  }
}
