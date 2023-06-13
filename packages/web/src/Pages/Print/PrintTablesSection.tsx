import { faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fGet } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { useChartData } from '../App/WithChartData'
import { useSimulation } from '../App/WithSimulation'
import { planChartLabel } from '../Plan/PlanChart/PlanChartMainCard/PlanChartLabel'
import { PlanChartType } from '../Plan/PlanChart/PlanChartType'
import { PrintSection } from './PrintSection'

export const PrintTablesSection = React.memo(() => {
  const { params } = useSimulation()
  const { extraSpending } = params.plan.adjustmentsToSpending

  const secondaryCharts: PlanChartType[] = _.compact([
    extraSpending.discretionary.length > 0 || extraSpending.essential.length > 0
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
  ])

  return (
    <>
      <PrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Appendix</h1>
      </PrintSection>
      <PrintSection>
        <_Table className="mt-10" type="spending-total" />
        {secondaryCharts.map((x, i) => (
          <_Table key={i} className="mt-10" type={x} />
        ))}
      </PrintSection>
    </>
  )
})

const _Table = React.memo(
  ({ className, type }: { className?: string; type: PlanChartType }) => {
    const { params, paramsExt } = useSimulation()
    const allChartData = useChartData()
    const chartMainData = fGet(allChartData.byYearsFromNowPercentiles.get(type))
    const { getCurrentAgeOfPerson } = paramsExt
    const person1Age = useMemo(
      () => getCurrentAgeOfPerson('person1'),
      [getCurrentAgeOfPerson],
    )
    const person2Age = useMemo(
      () =>
        params.plan.people.withPartner
          ? getCurrentAgeOfPerson('person2')
          : null,
      [getCurrentAgeOfPerson, params.plan.people.withPartner],
    )

    const months = _.range(
      chartMainData.months.displayRange.start,
      chartMainData.months.displayRange.end + 1,
    )
    const { label, subLabel, yAxisDescription } = planChartLabel(
      params,
      type,
      'full',
    )
    const yAxisDescriptionStr = yAxisDescription
      ? yAxisDescription.notMobile.map((x) => x.value).join(' ')
      : null

    return (
      <div className={clsx(className)}>
        <h2 className="flex items-center mt-4">
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
        {yAxisDescriptionStr && <h2 className="">{yAxisDescriptionStr}</h2>}
        <table className=" border-collapse mt-4 border border-black">
          <thead className="">
            <tr className="">
              <th className="px-4" colSpan={2}>
                Your Age
              </th>
              {person2Age !== null && (
                <th className="px-4 border-l border-black" colSpan={2}>
                  {`Partner's Age`}
                </th>
              )}
              <th className="px-4 border-l border-black" colSpan={3}>
                Percentiles
              </th>
            </tr>
            <tr className="border-b border-black">
              <th className="px-4">Years</th>
              <th className="px-4">Months</th>
              {person2Age !== null && (
                <>
                  <th className="px-4 border-l border-black">Years</th>
                  <th className="px-4">Months</th>
                </>
              )}
              <th className="px-4  border-l border-black">
                {chartMainData.percentiles[0].percentile}
                <span className=" align-super text-[8px]">th</span>
              </th>
              <th className="px-4">
                {chartMainData.percentiles[1].percentile}
                <span className=" align-super text-[8px]">th</span>
              </th>
              <th className="px-4">
                {chartMainData.percentiles[2].percentile}
                <span className=" align-super text-[8px]">th</span>
              </th>
            </tr>
          </thead>
          <tbody className="font-mono text-[10px]">
            {months.map((x, i) => {
              const month = (x + person1Age.inMonths) % 12
              if (!(month === 0 || i === months.length - 1 || i === 0))
                return <></>
              return (
                <tr key={i} className={clsx('border-t border-gray-400')}>
                  <td className="px-4 text-center">
                    {`${Math.floor((x + person1Age.inMonths) / 12)}`}{' '}
                  </td>
                  <td className="px-4 text-center">{`${month}`} </td>
                  {person2Age !== null && (
                    <>
                      <td className="px-4 text-center  border-l border-black">
                        {`${Math.floor((x + person2Age.inMonths) / 12)}`}{' '}
                      </td>
                      <td className="px-4 text-center">
                        {`${(x + person2Age.inMonths) % 12}`}{' '}
                      </td>
                    </>
                  )}
                  <td className="px-4 text-right  border-l border-black">
                    {chartMainData.yFormat(
                      chartMainData.percentiles[0].data(x),
                    )}{' '}
                  </td>
                  <td className="px-4 text-right">
                    {chartMainData.yFormat(
                      chartMainData.percentiles[1].data(x),
                    )}{' '}
                  </td>
                  <td className="px-4 text-right">
                    {chartMainData.yFormat(
                      chartMainData.percentiles[2].data(x),
                    )}{' '}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  },
)
