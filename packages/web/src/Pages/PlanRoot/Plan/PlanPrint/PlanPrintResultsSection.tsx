import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fGet } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { rectExt } from '../../../../Utils/Geometry'
import { getYAxisGridInfo } from '../../../Common/Chart/ChartComponent/ChartYAxis'
import { ChartReactStatefull } from '../../../Common/Chart/ChartReact'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanResultsChartType } from '../PlanResults/PlanResultsChartType'
import {
  planResultsLegacyCardFormat,
  usePlanResultsLegacyCardData,
} from '../PlanResults/PlanResultsSidePanel/PlanResultsSidePanelLegacyCard'
import { TPAWChartDataMain } from '../PlanResults/TPAWChart/TPAWChartDataMain'
import { tpawChartMainXAxisSizing } from '../PlanResults/TPAWChart/TPAWChartMain'
import { TPAWChartMainPrint } from '../PlanResults/TPAWChart/TPAWChartMainPrint'
import { useChartData } from '../WithChartData'
import { PlanPrintSection } from './PlanPrintSection'
import { planResultsChartLabel } from '../PlanResults/PlanResultsChartCard/PlanResultsChartLabel'
import { mainPlanColors } from '../UsePlanColors'

export const PlanPrintResultsSection = React.memo(() => {
  const { tpawResult, planParams } = useSimulation()
  const { extraSpending } = planParams.adjustmentsToSpending

  const secondaryCharts: PlanResultsChartType[][] = _.chunk(
    _.compact([
      _.values(extraSpending.discretionary).length > 0 ||
      _.values(extraSpending.essential).length > 0
        ? 'spending-general'
        : undefined,
      ..._.values(extraSpending.essential)
        .sort((a, b) => a.sortIndex - b.sortIndex)
        .map((x) => `spending-essential-${x.id}` as const),
      ..._.values(extraSpending.discretionary)
        .sort((a, b) => a.sortIndex - b.sortIndex)
        .map((x) => `spending-discretionary-${x.id}` as const),
      'portfolio' as const,
      'asset-allocation-savings-portfolio' as const,
      'withdrawal' as const,
    ]),
    3,
  )
  return (
    <>
      <PlanPrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Your Results</h1>
      </PlanPrintSection>
      <PlanPrintSection>
        <_Chart className="mt-8" type="spending-total" />
        {planParams.advanced.strategy === 'SWR' && (
          <h2 className=" mt-4 text-right">
            Success Rate:{' '}
            <span className="text-xl">
              {formatPercentage(0)(
                (tpawResult.numSimulationsActual -
                  tpawResult.numRunsWithInsufficientFunds) /
                  tpawResult.numSimulationsActual,
              )}
            </span>
          </h2>
        )}
        <_Legacy className="mt-12" />
      </PlanPrintSection>
      {secondaryCharts.map((x, i) => (
        <PlanPrintSection
          key={i}
          className={clsx(
            'flex flex-col',
            x.length === 3 ? 'justify-between' : 'justify-start gap-y-10',
          )}
        >
          {x.map((y, i) => (
            <_Chart key={i} className="" type={y} />
          ))}
        </PlanPrintSection>
      ))}
    </>
  )
})

const _Legacy = React.memo(({ className }: { className?: string }) => {
  const { planParams } = useSimulation()
  const data = usePlanResultsLegacyCardData()
  const maxLegacy = Math.max(...data.map((x) => x.amount))
  const hasLegacy =
    maxLegacy > 0 ||
    planParams.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0
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
                    {planResultsLegacyCardFormat(x.amount, 'desktop')}
                  </h2>
                </React.Fragment>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="">
              <span className="">$0</span>{' '}
              <div className="">(no legacy target entered)</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
})

const _Chart = React.memo(
  ({ className, type }: { className?: string; type: PlanResultsChartType }) => {
    const { planParams, tpawResult } = useSimulation()
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

    const { label, subLabel, yAxisDescription } = planResultsChartLabel(
      planParams,
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
            'relative   rounded-lg mt-2 ',

          )}
          style={{
            backgroundColor:mainPlanColors.results.print.chartBG
          }}
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
