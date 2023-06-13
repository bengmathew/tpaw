import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fGet } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
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
  const { params } = useSimulation()
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
    2,
  )
  return (
    <>
      <PrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Your Results</h1>
      </PrintSection>
      <PrintSection>
        <_Chart className="mt-8" type="spending-total" />
        <_Legacy className="mt-12" />
      </PrintSection>
      {secondaryCharts.map((x, i) => (
        <PrintSection key={i} className="flex flex-col justify-between">
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
  return (
    <div className={clsx(className)}>
      <h2 className="text-2xl font-bold mb-2">Legacy</h2>
      <div className="inline-block border-2 border-black rounded-lg p-2">
        {maxLegacy > 0 ||
        params.plan.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0 ? (
          <>
            <div
              className="inline-grid text-[14px] mt-0.5 gap-x-4 items-center "
              style={{ grid: 'auto/auto 1fr' }}
            >
              {data.map((x) => (
                <React.Fragment key={x.percentile}>
                  <h2 className="text-right text-lg">
                    {x.percentile}
                    <sup className="text-[10px]">th</sup> percentile
                  </h2>
                  <h2 className="font-mono text-[16px]">
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
              <span className="text-2xl font-bold">{x}</span>
            </React.Fragment>
          ))}
        </h2>
        {subLabel && <h2 className="text-xl font-bold">{subLabel}</h2>}
        {yAxisDescriptionStr && (
          <h2 className="text-base">
            {yAxisDescriptionStr}. The graph shows 5
            <span className=" align-super text-[10px]">th</span> to 95
            <span className=" align-super text-[10px]">th</span> percentiles
            with the 50<span className=" align-super text-[10px]">th</span>{' '}
            percentile in bold.
          </h2>
        )}

        <div
          className={clsx(
            // The border is applied to the div outside the measured div.
            'relative  border-[2px] border-black rounded-lg mt-2',
          )}
        >
          <div className="h-[325px]" ref={outerDivRef}>
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
      2 + // gap below plotArea.
      2 + // line at bottom of plotArea.
      2 + // gap below x axis.
      xAxisSizing.height +
      (hasPartner
        ? +3 * xAxisSizing.gap + xAxisSizing.height
        : 2 * xAxisSizing.gap),
  }

  return {
    position: rectExt({
      x: 0,
      y: 0,
      width,
      height: 325,
    }),
    padding,
  }
}
