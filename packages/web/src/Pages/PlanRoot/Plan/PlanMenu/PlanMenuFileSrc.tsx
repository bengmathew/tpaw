import { faCaretDown, faSave } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { assert } from '@tpaw/common'
import React, { useState } from 'react'
import { useUser } from '../../../App/WithUser'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import {
  SimulationInfoForFileSrc,
  SimulationInfoForPlanMode,
} from '../../PlanRootHelpers/WithSimulation'
import { PlanLocalStorage } from '../../PlanRootLocalMain/PlanLocalStorage'
import { usePlanColors } from '../UsePlanColors'
import { usePlanMenuSectionEditLocal } from './PlanMenuSection/PlanMenuSectionEditLocal'
import { usePlanMenuSectionOfflinePlans } from './PlanMenuSection/PlanMenuSectionOfflinePlans'
import { usePlanMenuSectionManagePlansLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansLoggedIn'
import { usePlanMenuSectionManagePlansNotLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansNotLoggedIn'
import { usePlanMenuSectionMisc } from './PlanMenuSection/PlanMenuSectionMisc'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'
import { usePlanMenuSectionTop } from './PlanMenuSection/PlanMenuSectionTop'

export const PlanMenuFileSrc = React.memo(
  ({
    simulationInfoForFileSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForFileSrc: SimulationInfoForFileSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const user = useUser()
    const planColors = usePlanColors()
    const { reset, plan, isModified, setForceNav } = simulationInfoForFileSrc

    const topSection = usePlanMenuSectionTop({
      simulationInfoBySrc: simulationInfoForFileSrc,
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
      info: { isCurrentlyFile: true, simulationInfoForFileSrc },
      simulationInfoForPlanMode,
    })
    const editSection = usePlanMenuSectionEditLocal({
      editLabelInfo: null,
      resetTo: {
        params: (planParams) => reset(planParams),
        // Intentionally not allowing "Undo Changes" for file plans.
        original: null,
      },
    })

    const fFilename = plan.filename ?? 'Unnamed'

    return (
      <div className="flex gap-x-2">
        <Menu>
          {({ open, close }) => (
            <ContextModal
              align="right"
              open={open}
              onOutsideClickOrEscape={null}
              afterLeave={() => offlinePlansSection.afterMenuClose()}
            >
              {({ ref }) => (
                <Menu.Button
                  ref={ref}
                  // m-0.5 to give space for focus ring
                  className="px-3 py-1.5 rounded-lg relative  m-0.5 "
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
                  <div
                    className="relative  flex items-center gap-x-2"
                    title={fFilename}
                  >
                    {/* Don't make this a flex box because that will remove ellipsis */}
                    <h2 className="max-w-[200px] overflow-hidden whitespace-nowrap text-ellipsis ">
                      {isModified && (
                        <span className=" text-white bg-errorBlockBG rounded-lg px-1.5 text-sm opacity-90 mr-1">
                          modified
                        </span>
                      )}
                      <span className="lighten-2 mr-1">File:</span>
                      {fFilename}
                    </h2>
                    <FontAwesomeIcon icon={faCaretDown} />
                  </div>
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
