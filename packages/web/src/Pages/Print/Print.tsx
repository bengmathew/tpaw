import { DateTime } from 'luxon'
import React from 'react'

import { faLeftLong, faPrint } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import { useURLUpdater } from '../../Utils/UseURLUpdater'
import { TasksForThisMonthContent } from '../TasksForThisMonth/TasksForThisMonth'
import { PrintInputSection } from './PrintInputSection'
import { PrintResultsSection } from './PrintResultsSection'
import { PrintSection } from './PrintSection'
import { PrintTablesSection } from './PrintTablesSection'

export const Print = React.memo(
  ({ className = '' }: { className?: string }) => {
    const urlUpdater = useURLUpdater()
    return (
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
            onClick={() => urlUpdater.push(_getPrintOnDoneURL())}
          >
            <FontAwesomeIcon className="" icon={faLeftLong} /> Done
          </button>
        </div>
        <button
          className=" print:hidden  fixed right-[50px] bottom-[50px] border-2 border-gray-500 p-1 rounded-full w-[150px] z-10"
          onClick={() => window.print()}
        >
          <div className="  btn-dark btn-lg flex items-center justify-center gap-x-4">
            <FontAwesomeIcon icon={faPrint} />
            Print
          </div>
        </button>
        <PrintSection className="flex flex-col relative">
          <div className=" ">
            <h1 className="font-bold text-[50px]">Retirement Plan</h1>
            <h2 className="font-bold text-[30px]">
              {DateTime.now().toLocaleString(DateTime.DATE_FULL)}
            </h2>
          </div>
          <div className="absolute bottom-[1.5cm] print:bottom-[.5cm] right-[1.5cm] flex flex-col items-end ">
            <h2 className="text-[25px] font-semibold ">TPAW Planner</h2>
            <h2 className="text-[18px] ">tpawplanner.com</h2>
          </div>
        </PrintSection>
        <PrintInputSection />
        <PrintResultsSection />
        <PrintSection className="flex items-center justify-center">
          <h1 className="font-bold text-4xl text-center ">
            Tasks for This Month
          </h1>
        </PrintSection>
        <PrintSection>
          <TasksForThisMonthContent className="mt-10" />
        </PrintSection>
        <PrintTablesSection />
      </div>
    )
  },
)

export const setPrintOnDoneURL = (url: string) => {
  window.localStorage.setItem('PrintOnDoneURL', url)
}
const _getPrintOnDoneURL = (): string => {
  const result = window.localStorage.getItem('PrintOnDoneURL') as
    | string
    | undefined

  window.localStorage.removeItem('PrintOnDoneURL')
  return result ?? '/plan'
}
