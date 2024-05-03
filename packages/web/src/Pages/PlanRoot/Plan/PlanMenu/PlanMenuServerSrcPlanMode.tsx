import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import React from 'react'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import {
  SimulationInfoForPlanMode,
  SimulationInfoForServerSrc,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { usePlanMenuSectionEditServer } from './PlanMenuSection/PlanMenuSectionEditServer'
import { usePlanMenuSectionFile } from './PlanMenuSection/PlanMenuSectionFile'
import { usePlanMenuSectionManagePlansLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansLoggedIn'
import { usePlanMenuSectionMisc } from './PlanMenuSection/PlanMenuSectionMisc'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'

export const PlanMenuServerSrcPlanMode = React.memo(
  ({
    simulationInfoForServerSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForServerSrc: SimulationInfoForServerSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()
    const { plan } = simulationInfoForServerSrc
    const label = plan.isMain ? 'Main Plan' : plan.label ?? 'Untitled'

    const managePlansSection = usePlanMenuSectionManagePlansLoggedIn({
      simulationInfoForServerSrc,
      isModified: false,
    })
    const miscSection = usePlanMenuSectionMisc({
      simulationInfoForServerSrc,
    })
    const fileSection = usePlanMenuSectionFile({
      info: { isCurrentlyFile: false, label: plan.label ?? null },
      simulationInfoForPlanMode,
    })
    const editSection = usePlanMenuSectionEditServer({
      simulationInfoForServerSrc,
    })
    return (
      <div className="flex gap-x-2">
        <Menu>
          {({ open }) => (
            <ContextModal
              align="right"
              open={open}
              onOutsideClickOrEscape={null}
              afterLeave={() => fileSection.afterMenuClose()}
            >
              {({ ref }) => (
                <Menu.Button
                  ref={ref}
                  // m-0.5 gives space for focus ring
                  className="px-3 py-1.5 rounded-lg m-0.5 "
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
                  <div
                    className="relative  flex items-center gap-x-2"
                    title={label}
                  >
                    <h2 className="max-w-[200px] overflow-hidden whitespace-nowrap text-ellipsis">
                      {label}
                    </h2>
                    <FontAwesomeIcon icon={faCaretDown} />
                  </div>
                </Menu.Button>
              )}
              <Menu.Items className="flex flex-col context-menu-outer-div min-w-[275px]">
                {managePlansSection.menuItems}
                {miscSection.menuItems}
                {fileSection.menuItems}
                {editSection.menuItems}
              </Menu.Items>
            </ContextModal>
          )}
        </Menu>
        <PlanMenuSubMenuUndoRedo
          simulationDetailForPlanMode={simulationInfoForPlanMode}
          className={{ undo: 'pl-6 pr-3 ', redo: 'pl-3 pr-6 ' }}
        />
        {managePlansSection.modals}
        {miscSection.modals}
        {fileSection.modals}
        {editSection.modals}
      </div>
    )
  },
)
