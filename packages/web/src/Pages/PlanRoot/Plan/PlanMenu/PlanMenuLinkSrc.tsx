import { faCaretDown, faSave } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import React, { useState } from 'react'
import { useUser } from '../../../App/WithUser'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import {
  SimulationInfoForLinkSrc,
  SimulationInfoForPlanMode,
} from '../../PlanRootHelpers/WithSimulation'
import { PlanLocalStorage } from '../../PlanRootLocalMain/PlanLocalStorage'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionModalLoginAndSaveLink } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLoginAndSaveWorkingPlan'
import { PlanMenuActionModalOverwriteLocal } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalOverwriteLocal'
import { PlanMenuActionModalSaveWorkingPlanToAccount } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSaveWorkingPlanToAccount'
import { usePlanMenuSectionEditLocal } from './PlanMenuSection/PlanMenuSectionEditLocal'
import { usePlanMenuSectionOfflinePlans } from './PlanMenuSection/PlanMenuSectionOfflinePlans'
import { usePlanMenuSectionManagePlansLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansLoggedIn'
import { usePlanMenuSectionManagePlansNotLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansNotLoggedIn'
import { usePlanMenuSectionMisc } from './PlanMenuSection/PlanMenuSectionMisc'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'
import { assert } from '@tpaw/common'
import { usePlanMenuSectionTop } from './PlanMenuSection/PlanMenuSectionTop'

export const PlanMenuLinkSrc = React.memo(
  ({
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()
    const user = useUser()

    const { isModified, reset, setForceNav } = simulationInfoForLinkSrc

    const topSection = usePlanMenuSectionTop({
      simulationInfoBySrc: simulationInfoForLinkSrc,
      simulationInfoForPlanMode,
    })
    const managePlansSection = user
      ? // eslint-disable-next-line react-hooks/rules-of-hooks
        usePlanMenuSectionManagePlansLoggedIn({
          simulationInfoForServerSrc: null,
          isModified: isModified ? { setForceNav } : false,
        })
      : // eslint-disable-next-line react-hooks/rules-of-hooks
        usePlanMenuSectionManagePlansNotLoggedIn()

    const miscSection = usePlanMenuSectionMisc({
      simulationInfoForServerSrc: null,
    })
    const offlinePlansSection = usePlanMenuSectionOfflinePlans({
      info: { isCurrentlyFile: false, label: null },
      simulationInfoForPlanMode,
    })
    const editSection = usePlanMenuSectionEditLocal({
      editLabelInfo: null,
      resetTo: {
        params: (planParams) => reset(planParams),
        original: { isModified, reset: () => reset(null) },
      },
    })

    return (
      <div className="flex gap-x-2">
        <Menu>
          {({ open, close }) => (
            <ContextModal
              align="right"
              open={open}
              onOutsideClickOrEscape={null}
            >
              {({ ref }) => (
                <Menu.Button
                  ref={ref}
                  className="px-3 py-1.5 rounded-lg relative flex  items-center  m-0.5"
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
                  <h2 className="text-start flex items-center">
                    {isModified && (
                      <span className=" text-white bg-errorBlockBG rounded-lg px-1.5 text-sm opacity-90 mr-1">
                        modified
                      </span>
                    )}
                    Plan From Link{' '}
                  </h2>
                  <FontAwesomeIcon className="ml-2" icon={faCaretDown} />
                </Menu.Button>
              )}
              <Menu.Items className="flex flex-col context-menu-outer-div w-[275px]">
                {topSection.menuItems}
                {managePlansSection.menuItems}
                {miscSection.menuItems}
                {offlinePlansSection.menuItems}
                {editSection.menuItems}
              </Menu.Items>
            </ContextModal>
          )}
        </Menu>
        <PlanMenuSubMenuUndoRedo
          simulationDetailForPlanMode={simulationInfoForPlanMode}
          className={{ undo: 'pl-6 pr-3', redo: 'pl-3 pr-6' }}
        />

        {topSection.modals}
        {managePlansSection.modals}
        {miscSection.modals}
        {offlinePlansSection.modals}
        {editSection.modals}
      </div>
    )
  },
)
