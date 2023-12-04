import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, fGet } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { getChartBandColor } from '../../../../Utils/ColorPalette'
import { RGB } from '../../../../Utils/ColorUtils'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { rectExt } from '../../../../Utils/Geometry'
import { interpolate } from '../../../../Utils/Interpolate'
import { ChartReactStatefull } from '../../../Common/Chart/ChartReact'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanResultsChartData } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartData'
import { PlanResultsChartType } from '../PlanResults/PlanResultsChartType'
import {
  planResultsLegacyCardFormat,
  usePlanResultsLegacyCardData,
} from '../PlanResults/PlanResultsSidePanel/PlanResultsSidePanelLegacyCard'
import { PlanColors, mainPlanColors, usePlanColors } from '../UsePlanColors'
import { useChartData } from '../WithPlanResultsChartData'
import { getPlanPrintChartLabel } from './GetPlanPrintChartLabel'
import {
  PlanPrintChart,
  planPrintChartOutsidePlotAreaSizing,
} from './PlanPrintChart'
import { PlanPrintSVGBackground } from './PlanPrintSVGBackground'
import { PlanPrintSection } from './PlanPrintSection'
import { getChartBreakdownTotalFillPattern } from '../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartBreakdown'

export const PlanPrintResultsSection = React.memo(() => {
  const { tpawResult, planParams } = useSimulation()
  const { extraSpending } = planParams.adjustmentsToSpending

  const secondaryCharts: PlanResultsChartType[] = _.compact([
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
  ])

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
        {_.keys(planParams.wealth.incomeDuringRetirement).length > 0 && (
          <div className=" break-inside-avoid-page">
            <_Chart className="mt-12" type="spending-total-funding-sources-5" />
            <_MonthlySpendingBreakdownLegend className="" />
          </div>
        )}
        {secondaryCharts.map((x, i) => (
          <_Chart key={i} className="mt-12" type={x} />
        ))}
      </PlanPrintSection>
    </>
  )
})

const _MonthlySpendingBreakdownLegend = React.memo(
  ({ className }: { className?: string }) => {
    const planColors = usePlanColors()
    const chartData = useChartData('spending-total-funding-sources-5')
    assert(chartData.type === 'breakdown')
    const parts = [
      ...chartData.breakdown.parts
        .slice()
        .reverse()
        .map((part) => ({
          fillPattern: getChartBandColor(part.chartColorIndex).fillPattern,
          label: part.label,
        })),
      {
        fillPattern: getChartBreakdownTotalFillPattern(planColors),
        label: 'From Portfolio',
      },
    ]
    const cols = _.chunk(parts, Math.ceil(parts.length / 3))
    return (
      // Make sure
      <div className={clix(className)}>
        <div
          className="inline-grid gap-x-4"
          style={{ grid: `auto/${_.repeat('1fr ', cols.length)}` }}
        >
          {cols.map((col, i) => (
            <div key={i} className="">
              {col.map(({ fillPattern, label }, i) => {
                return (
                  <div key={i} className="flex gap-x-2 items-start">
                    <div
                      className="w-[10px] h-[10px] relative rounded-sm overflow-hidden shrink-0 mt-1"
                      style={{ opacity: `${fillPattern.bg.opacity}` }}
                    >
                      <PlanPrintSVGBackground
                        className=""
                        fill={RGB.toHex(
                          interpolate({
                            from: fillPattern.bg.color,
                            target: fillPattern.stroke.color,
                            progress: 0.7,
                          }),
                        )}
                      />
                    </div>
                    <h2>{label ?? '<No label>'}</h2>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    )
  },
)

const _Legacy = React.memo(({ className }: { className?: string }) => {
  const { planParams } = useSimulation()
  const data = usePlanResultsLegacyCardData()
  const maxLegacy = Math.max(...data.map((x) => x.amount))
  const hasLegacy =
    maxLegacy > 0 ||
    planParams.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0
  return (
    <div className={clix(className)}>
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
    const chartData = useChartData(type)
    const outerDivRef = useRef<HTMLDivElement>(null)

    const hasPartner = tpawResult.params.people.withPartner
    const [chart, setChart] = useState<ChartReactStatefull<{
      data: PlanResultsChartData
    }> | null>(null)
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

    const { label, subLabel, yAxisDescriptionStr, description } =
      getPlanPrintChartLabel(planParams, type)

    return (
      <div className={clix(className, ' break-inside-avoid-page')}>
        <h2 className="flex items-center">
          {label.full.map((x, i) => (
            <React.Fragment key={i}>
              {i > 0 && (
                <FontAwesomeIcon className="mx-3" icon={faChevronRight} />
              )}
              <span className="text-xl font-bold">{x}</span>
            </React.Fragment>
          ))}
        </h2>
        {subLabel && <h2 className="text-lg font-bold">{subLabel}</h2>}
        <h2 className="text-[11px]">
          {yAxisDescriptionStr && (
            <span className="">{yAxisDescriptionStr}. </span>
          )}
          {description}
        </h2>

        <div
          className={clix(
            // The border is applied to the div outside the measured div.
            'relative   rounded-lg mt-2 overflow-hidden ',
          )}
        >
          <div className="relative h-[225px]" ref={outerDivRef}>
            <PlanPrintSVGBackground
              className=""
              fill={mainPlanColors.shades.alt[2].hex}
            />
            <PlanPrintChart
              starting={{
                data: chartData,
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
  const padding = {
    left: 90,
    right: 10,
    top: 20, // 20 is needed for the max y label.
    bottom:
      planPrintChartOutsidePlotAreaSizing.below.gapToLine +
      planPrintChartOutsidePlotAreaSizing.below.lineWidth +
      planPrintChartOutsidePlotAreaSizing.below.xAxis.height +
      (hasPartner
        ? planPrintChartOutsidePlotAreaSizing.below.xAxis.vertGapBetweenPeople +
          planPrintChartOutsidePlotAreaSizing.below.xAxis.height
        : 0) +
      planPrintChartOutsidePlotAreaSizing.below.xAxis.vertGapBetweenPeople * 2,
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
