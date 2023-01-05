import _ from 'lodash'
import React, { CSSProperties, useMemo } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { useSimulation } from '../../../App/WithSimulation'

export const PlanChartSidePanelLegacyCard = React.memo(
  ({
    className = '',
    style,
    layout,
  }: {
    className?: string
    style: CSSProperties
    layout: 'laptop' | 'desktop' | 'mobile'
  }) => {
    const { params } = useSimulation()
    const data = usePlanChartLegacyCardData()
    const maxLegacy = Math.max(...data.map((x) => x.amount))

    return (
      <div className={`${className} `} style={style}>
        <h2 className="font-bold text-[16px] sm:text-[22px]">Legacy</h2>
        {maxLegacy > 0 ||
        params.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0 ? (
          <>
            <h2 className=" text-xs sm:text-sm border-b border-gray-300 mt-1 ">
              Percentiles
            </h2>
            <div
              className="grid font-mono text-[12px] mt-0.5 gap-x-4"
              style={{
                grid: 'auto/auto 1fr',
                lineHeight: layout === 'mobile' ? '16px' : '18px',
              }}
            >
              {data.map((x) => (
                <React.Fragment key={x.percentile}>
                  <h2 className="text-right">
                    {x.percentile}
                    <sup className="text-[8px]">th</sup>
                  </h2>
                  <h2 className="text-right">
                    {planChartLegacyCardFormat(x.amount, layout)}
                  </h2>
                </React.Fragment>
              ))}
            </div>
          </>
        ) : (
          <div className="mt-2 text-center ">
            $0 <div className="lighten-2">No legacy target entered</div>
          </div>
        )}
      </div>
    )
  },
)

export function usePlanChartLegacyCardData() {
  const simulation = useSimulation()
  return useMemo(() => {
    const { tpawResult, highlightPercentiles } = simulation
    const { endingBalanceOfSavingsPortfolioByPercentile, args } = tpawResult
    return _.sortBy(
      endingBalanceOfSavingsPortfolioByPercentile
        .filter((x) => highlightPercentiles.includes(x.percentile))
        .map((x) => ({
          amount:
            x.data +
            args.params.adjustmentsToSpending.tpawAndSPAW.legacy.external,
          percentile: x.percentile,
        })),
      (x) => -x.percentile,
    )
  }, [simulation])
}

export const planChartLegacyCardFormat = (
  x: number,
  layout: 'laptop' | 'desktop' | 'mobile',
) => {
  if (layout !== 'mobile') return formatCurrency(x)
  return x < 1000
    ? `${formatCurrency(x)}`
    : x < 1000000
    ? `${formatCurrency(Math.round(x / 1000))}K`
    : `${formatCurrency(_.round(x / 1000000, 1), 1)}M`
}
