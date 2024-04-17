import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { CalendarMonthFns, block, fGet, noCase } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React from 'react'
import { PERCENTILES_STR } from '../../../../UseSimulator/Simulator/Simulator'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { PlanResultsChartType } from '../../Plan/PlanResults/PlanResultsChartType'
import { useChartDataForPDF } from '../../Plan/WithPlanResultsChartData'
import { useSimulationResult } from '../WithSimulation'
import { getPlanPrintChartLabel } from './Helpers/GetPlanPrintChartLabel'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'
import { InMonthsFns } from '../../../../Utils/InMonthsFns'

export const PlanPrintViewAppendixSection = React.memo(
  ({ settings }: { settings: PlanPrintViewArgs['settings'] }) => {
    const { args } = useSimulationResult()
    const { extraSpending } = args.planParamsNorm.adjustmentsToSpending

    const secondaryCharts: PlanResultsChartType[] = _.compact([
      extraSpending.discretionary.length > 0 ||
      extraSpending.essential.length > 0
        ? 'spending-general'
        : undefined,
      ...extraSpending.essential
        .sort((a, b) => a.sortIndex - b.sortIndex)
        .map((x) => `spending-essential-${x.id}` as const),
      ...extraSpending.discretionary
        .sort((a, b) => a.sortIndex - b.sortIndex)
        .map((x) => `spending-discretionary-${x.id}` as const),
      'portfolio' as const,
      'asset-allocation-savings-portfolio' as const,
      'withdrawal' as const,
    ])

    return (
      <>
        <PlanPrintViewPageGroup
          className="flex items-center justify-center"
          settings={settings}
        >
          <h1 className="font-bold text-4xl text-center ">Appendix</h1>
        </PlanPrintViewPageGroup>
        <PlanPrintViewPageGroup settings={settings}>
          <_Table className="mt-10" type="spending-total" />
          {args.planParamsNorm.wealth.incomeDuringRetirement.length > 0 && (
            <_Table
              className="mt-10"
              type="spending-total-funding-sources-50"
            />
          )}
          {secondaryCharts.map((x, i) => (
            <_Table key={i} className="mt-10" type={x} />
          ))}
        </PlanPrintViewPageGroup>
      </>
    )
  },
)

const _Table = React.memo(
  ({ className, type }: { className?: string; type: PlanResultsChartType }) => {
    const { args } = useSimulationResult()
    const chartData = useChartDataForPDF(type)
    const { ages } = args.planParamsNorm

    const months = _.range(
      chartData.displayRange.x.start,
      chartData.displayRange.x.end + 1,
    )
    const { label, subLabel, yAxisDescriptionStr } = getPlanPrintChartLabel(
      args.planParamsNorm,
      type,
    )

    return (
      <div className={clix(className)}>
        <h2 className="flex items-center mt-4">
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
        {yAxisDescriptionStr && <h2 className="">{yAxisDescriptionStr}</h2>}
        <table className=" border-collapse mt-2 border border-black">
          <thead className="">
            <tr className="">
              {args.planParamsNorm.datingInfo.isDated && (
                <th className="px-4 border-l border-black" rowSpan={2}>
                  Month
                </th>
              )}
              <th className="px-4 border-l border-black" rowSpan={2}>
                Your Age
              </th>
              {ages.person2 !== null && (
                <th className="px-4 border-l border-black" rowSpan={2}>
                  {`Partner's Age`}
                </th>
              )}
              {chartData.type === 'range' ? (
                <th className="px-4 border-l border-black" colSpan={3}>
                  Percentiles
                </th>
              ) : chartData.type === 'breakdown' ? (
                <>
                  <th
                    className="px-4 border-l border-black text-right"
                    rowSpan={2}
                  >
                    Income <br /> During <br /> Retirement
                  </th>
                  <th className="px-2">+</th>
                  <th className="px-4 text-right " rowSpan={2}>
                    Withdrawal <br />
                    From <br />
                    Portfolio
                  </th>
                  <th className="px-2">=</th>
                  <th className="px-4 text-right" rowSpan={2}>
                    Total <br />
                    Spending
                  </th>
                </>
              ) : (
                noCase(chartData)
              )}
            </tr>
            <tr className="">
              {chartData.type === 'range' ? (
                <>
                  <th className="px-4  border-l border-black">
                    {PERCENTILES_STR[0]}
                    <span className=" align-super text-[8px]">th</span>
                  </th>
                  <th className="px-4">
                    {PERCENTILES_STR[1]}
                    <span className=" align-super text-[8px]">th</span>
                  </th>
                  <th className="px-4">
                    {PERCENTILES_STR[2]}
                    <span className=" align-super text-[8px]">th</span>
                  </th>
                </>
              ) : chartData.type === 'breakdown' ? (
                <></>
              ) : (
                noCase(chartData)
              )}
            </tr>
          </thead>
          <tbody className="font-mono text-[10px] ">
            {months.map((mfn, i) => {
              const shouldShow =
                InMonthsFns.getFromMFN(ages.person1.currentAgeInfo, 0, 1)(mfn) %
                  12 ===
                  0 ||
                i === months.length - 1 ||
                i === 0
              if (!shouldShow) return <React.Fragment key={i}></React.Fragment>

              const calendarMonth = args.planParamsNorm.datingInfo
                .nowAsCalendarMonth
                ? CalendarMonthFns.getFromMFN(
                    args.planParamsNorm.datingInfo.nowAsCalendarMonth,
                  )(mfn)
                : null
              return (
                <tr
                  key={i}
                  className={clix(
                    i === 0
                      ? 'border-t border-black'
                      : 'border-t border-gray-300',
                  )}
                >
                  {calendarMonth && (
                    <td className="px-4 text-center border-l border-black">
                      {CalendarMonthFns.toStr(calendarMonth, {
                        shortMonth: true,
                      })}
                    </td>
                  )}
                  <td className="px-4 text-center border-l border-black">
                    {`${Math.floor((mfn + ages.person1.currentAgeInfo.inMonths) / 12)}`}
                    <span className="ml-1 lighten-2 text-[8px]">yr</span>{' '}
                    <span>
                      {(mfn + ages.person1.currentAgeInfo.inMonths) % 12}
                    </span>
                    <span className="ml-1 lighten-2 text-[8px]">mo</span>
                  </td>
                  {ages.person2 !== null && (
                    <td className="px-4 text-center border-l border-black">
                      {`${Math.floor((mfn + ages.person2.currentAgeInfo.inMonths) / 12)}`}
                      <span className="ml-1 lighten-2 text-[8px]">yr</span>{' '}
                      <span>
                        {(mfn + ages.person2.currentAgeInfo.inMonths) % 12}
                      </span>
                      <span className="ml-1 lighten-2 text-[8px]">mo</span>
                    </td>
                  )}
                  {chartData.type === 'range'
                    ? block(() => {
                        const yRange = SimpleRange.Closed.isIn(
                          mfn,
                          chartData.range.xRange,
                        )
                          ? fGet(chartData.range.yRangeByX[mfn])
                          : null
                        return (
                          <>
                            <td className="px-4 text-right  border-l border-black">
                              {yRange ? chartData.formatY(yRange.start) : '-'}{' '}
                            </td>
                            <td className="px-4 text-right">
                              {yRange ? chartData.formatY(yRange.mid) : '-'}{' '}
                            </td>
                            <td className="px-4 text-right">
                              {yRange ? chartData.formatY(yRange.end) : '-'}{' '}
                            </td>
                          </>
                        )
                      })
                    : chartData.type === 'breakdown'
                      ? block(() => {
                          const checkedGet = ({
                            xRange,
                            yByX,
                          }: {
                            xRange: SimpleRange | null
                            yByX: (number | null)[] | Float64Array
                          }) =>
                            SimpleRange.Closed.isIn(mfn, xRange)
                              ? fGet(yByX[mfn])
                              : null
                          const fromIncome = chartData.breakdown.parts
                            .map((x) => checkedGet(x.data))
                            .reduce(
                              (a, b) =>
                                a === null && b === null
                                  ? null
                                  : (a ?? 0) + (b ?? 0),
                              null,
                            )
                          const total = checkedGet(chartData.breakdown.total)
                          const fromPortfolio =
                            total === null ? null : total - (fromIncome ?? 0)
                          return (
                            <>
                              <td className="px-4 text-right  border-l border-black">
                                {fromIncome !== null
                                  ? chartData.formatY(fromIncome)
                                  : '—'}
                              </td>
                              <td className="text-center lighten-2">+</td>
                              <td className="px-4 text-right">
                                {fromPortfolio !== null
                                  ? chartData.formatY(fromPortfolio)
                                  : '—'}
                              </td>
                              <td className="text-center lighten-2">=</td>
                              <td className="px-4 text-right">
                                {total !== null
                                  ? chartData.formatY(total)
                                  : '—'}
                              </td>
                            </>
                          )
                        })
                      : noCase(chartData)}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  },
)
