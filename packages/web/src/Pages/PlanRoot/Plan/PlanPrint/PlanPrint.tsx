import { DateTime } from 'luxon'
import React from 'react'

import { faLeftLong, faPrint } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import Head from 'next/head'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import {
  SimulationInfo,
  useSimulation,
} from '../../PlanRootHelpers/WithSimulation'
import { BalanceSheetContent } from '../PlanResults/PlanResultsSidePanel/PlanResultsSidePanelMenu/PlanResutlsSidePanelMenuBalanceSheet'
import { TasksForThisMonthContent } from '../PlanResults/PlanResultsSidePanel/PlanResultsSidePanelTasksCard'
import { PlanPrintInputSection } from './PlanPrintInputSection'
import { PlanPrintResultsSection } from './PlanPrintResultsSection'
import { PlanPrintSection } from './PlanPrintSection'
import { PrintTablesSection } from './PrintTablesSection'

export const PlanPrint = React.memo(
  ({
    className = '',
    planLabel,
  }: {
    className?: string
    planLabel: string | null
  }) => {
    const { planPaths } = useSimulation()
    const urlUpdater = useURLUpdater()
    return (
      <>
        <Head>
          <title>{`Plan ${
            planLabel ? `(${planLabel})` : ''
          }- Print - TPAW Planner'`}</title>
        </Head>
        <div
          className={clsx(
            className,
            'bg-gray-300 print:bg-white',
            'pb-10 print:pb-0',
            `font-font1 text-black flex flex-col items-center `,
          )}
        >
          <div className="print:hidden flex items-start w-[21cm]  sticky top-5 z-10">
            <button
              className="btn-dark btn-md "
              onClick={() => urlUpdater.push(_getPrintOnDoneURL(planPaths))}
            >
              <FontAwesomeIcon className="" icon={faLeftLong} /> Done
            </button>
          </div>
          <button
            className=" print:hidden  fixed right-[50px] bottom-[50px] border-2 border-gray-500 p-1 rounded-full w-[275px] z-10"
            onClick={() => window.print()}
          >
            <div className="  btn-dark btn-lg flex items-center justify-center gap-x-4">
              <FontAwesomeIcon icon={faPrint} />
              Print / Save as PDF
            </div>
          </button>
          <PlanPrintSection className="flex flex-col relative">
            <div className="">
              <h2 className="font-bold text-[50px] leading-[50px]">
                {DateTime.now().toFormat('yyyy')}
              </h2>
              <h2 className="font-bold text-[25px] leading-[25px] ">
                {DateTime.now().toFormat('MMMM dd')}
              </h2>
            </div>
            <div className=" ">
              <h1 className="font-bold text-[65px] mt-[105px] leading-[65px]">
                Retirement <br /> Plan
              </h1>
            </div>
            <div className="absolute bottom-[1.5cm] print:bottom-[.5cm] right-[1.5cm] flex flex-col items-end ">
              <h2 className="text-[25px] font-semibold ">TPAW Planner</h2>
              <h2 className="text-[18px] ">tpawplanner.com</h2>
            </div>
          </PlanPrintSection>
          <PlanPrintInputSection />
          <PlanPrintResultsSection />
          <PlanPrintSection className="flex items-center justify-center">
            <h1 className="font-bold text-4xl text-center ">
              Tasks for This Month
            </h1>
          </PlanPrintSection>
          <PlanPrintSection>
            <TasksForThisMonthContent className="mt-10" forPrint />
          </PlanPrintSection>
          <PlanPrintSection className="flex items-center justify-center">
            <h1 className="font-bold text-4xl text-center ">Balance Sheet</h1>
          </PlanPrintSection>
          <PlanPrintSection>
            <BalanceSheetContent forPrint />
          </PlanPrintSection>
          <PrintTablesSection />
        </div>
      </>
    )
  },
)

export const setPrintOnDoneURL = () => {
  window.localStorage.setItem('PlanPrint_onDoneURL', window.location.href)
}
const _getPrintOnDoneURL = (planPaths: SimulationInfo['planPaths']): URL => {
  const result = window.localStorage.getItem('PlanPrint_onDoneURL') as
    | string
    | undefined

  window.localStorage.removeItem('PlanPrint_onDoneURL')
  return result ? new URL(result) : planPaths()
}
