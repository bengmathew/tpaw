import React from 'react'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { BalanceSheetContent } from '../../Plan/PlanResults/PlanResultsSidePanel/PlanResultsSidePanelMenu/PlanResultsSidePanelMenuBalanceSheet'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'

export const PlanPrintViewBalanceSheetSection = React.memo(
  ({ settings }: { settings: PlanPrintViewArgs['settings'] }) => {
    return (
      <>
        <PlanPrintViewPageGroup
          className="flex items-center justify-center"
          settings={settings}
        >
          <h1 className="font-bold text-4xl text-center ">Balance Sheet</h1>
        </PlanPrintViewPageGroup>
        <PlanPrintViewPageGroup settings={settings}>
          <BalanceSheetContent forPrint />
        </PlanPrintViewPageGroup>
      </>
    )
  },
)
