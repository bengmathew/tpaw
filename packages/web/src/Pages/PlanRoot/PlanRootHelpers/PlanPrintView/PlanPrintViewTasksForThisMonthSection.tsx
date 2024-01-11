import React from 'react'
import { TasksForThisMonthContent } from '../../Plan/PlanResults/PlanResultsSidePanel/PlanResultsSidePanelTasksCard'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'

export const PlanPrintViewTasksForThisMonthSection = React.memo(
  ({ settings }: { settings: PlanPrintViewArgs['settings'] }) => {
    return (
      <>
        <PlanPrintViewPageGroup
          className="flex items-center justify-center"
          settings={settings}
        >
          <h1 className="font-bold text-4xl text-center ">
            Tasks for This Month
          </h1>
        </PlanPrintViewPageGroup>
        <PlanPrintViewPageGroup settings={settings}>
          <TasksForThisMonthContent forPrint />
        </PlanPrintViewPageGroup>
      </>
    )
  },
)
