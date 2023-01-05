import Link from 'next/link'
import React, { CSSProperties } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { Padding } from '../../../../Utils/Geometry'
import { useSimulation } from '../../../App/WithSimulation'
import { getTasksForThisYearProps, setTasksForThisYearOnDoneSection } from '../../../TasksForThisYear/TasksForThisYear'
import { PlanSectionName } from '../../PlanInput/Helpers/PlanSectionName'

export const PlanChartSidePanelTasksCard = React.memo(
  ({
    className = '',
    style,
    cardPadding,
    layout,
    section
  }: {
    className?: string
    style?: CSSProperties
    cardPadding: Padding
    layout: 'laptop' | 'desktop' | 'mobile'
    section:PlanSectionName
  }) => {
    const { tpawResult } = useSimulation()
    const { contributionToOrWithdrawalFromSavingsPortfolio, afterWithdrawals } =
      getTasksForThisYearProps(tpawResult)
    const withdrawOrContribute = (() =>
      contributionToOrWithdrawalFromSavingsPortfolio.type === 'withdrawal'
        ? {
            text: 'Withdraw',
            amount: contributionToOrWithdrawalFromSavingsPortfolio.withdrawal,
          }
        : {
            text: 'Contribute',
            amount: contributionToOrWithdrawalFromSavingsPortfolio.contribution,
          })()
    return (
      <Link href="/plan/tasks-for-this-year" shallow>
        <a
          className={`${className}  bg-orange-100 block overflow-hidden`}
          style={style}
          onClick={() => {
            setTasksForThisYearOnDoneSection(section)
          }}
        >
          <h2 className="font-bold text-[16px] sm:text-[22px]">Tasks</h2>

          <h2 className="font-medium text-[13px] sm:text-[15px] mt-1">
            {withdrawOrContribute.text}
          </h2>
          <h2 className=" text-[13px]">
            {formatCurrency(withdrawOrContribute.amount)}
          </h2>

          {layout !== 'mobile' && (
            <>
              <h2 className="font-medium text-[13px] sm:text-[15px] mt-1">
                Rebalance
              </h2>
              <div className="grid" style={{ grid: 'auto/auto auto' }}>
                <h2 className="text-[13px]">Stocks</h2>
                <h2 className="text-[13px] text-right">
                  {formatPercentage(0)(afterWithdrawals.allocation.stocks)}
                </h2>
                <h2 className="text-[13px]">Bonds</h2>
                <h2 className="text-[13px] text-right">
                  {formatPercentage(0)(1 - afterWithdrawals.allocation.stocks)}
                </h2>
              </div>
            </>
          )}
          <div
            className="flex justify-end mt-2"
            style={{
              marginRight: `${-cardPadding.right}px`,
              marginBottom: `${-cardPadding.bottom}px`,
            }}
          >
            <h2 className="bg-gray-700 text-white py-0.5 pl-4 pr-4 rounded-tl-lg">
              More
            </h2>
          </div>
        </a>
      </Link>
    )
  },
)
