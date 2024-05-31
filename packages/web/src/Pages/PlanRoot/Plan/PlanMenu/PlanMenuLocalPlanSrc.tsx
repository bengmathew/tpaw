import { faCaretDown, faUser } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import Link from 'next/link'
import React, { useEffect, useState } from 'react'
import { appPaths } from '../../../../AppPaths'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import {
  SimulationInfoForLocalMainSrc,
  SimulationInfoForPlanMode,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { usePlanMenuSectionEditLocal } from './PlanMenuSection/PlanMenuSectionEditLocal'
import { usePlanMenuSectionOfflinePlans } from './PlanMenuSection/PlanMenuSectionOfflinePlans'
import { usePlanMenuSectionManagePlansNotLoggedIn } from './PlanMenuSection/PlanMenuSectionManagePlansNotLoggedIn'
import { usePlanMenuSectionMisc } from './PlanMenuSection/PlanMenuSectionMisc'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'

export const PlanMenuLocalSrc = React.memo(
  ({
    simulationInfoForLocalMainSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForLocalMainSrc: SimulationInfoForLocalMainSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()
    const { reset } = simulationInfoForLocalMainSrc

    const [loginModalState, setLoginModalState] = useState<{
      heading: string
      message: string
    } | null>(null)

    const managePlansSection = usePlanMenuSectionManagePlansNotLoggedIn()
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
        original: null,
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
                  className="px-3 py-1.5 rounded-lg relative m-0.5 "
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
                  <h2 className="text-start font-semibold">Guest Plan</h2>
                  <div className="flex items-center">
                    <h2 className="mt-1 max-w-[200px] whitespace-nowrap text-ellipsis overflow-hidden ">
                      Saved on Browser
                    </h2>
                    <FontAwesomeIcon
                      className=" -mb-1 ml-2"
                      icon={faCaretDown}
                    />
                  </div>
                </Menu.Button>
              )}
              <Menu.Items className="flex flex-col context-menu-outer-div w-[275px]">
                <Menu.Item>
                  <Link
                    className={'context-menu-item '}
                    href={appPaths.login(appPaths.plan())}
                  >
                    <span className="context-menu-icon ">
                      <FontAwesomeIcon className="" icon={faUser} />
                    </span>{' '}
                    Save Plan to Account
                  </Link>
                </Menu.Item>

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

        <_LoginRequiredModal
          state={loginModalState}
          onHide={() => setLoginModalState(null)}
        />
        {managePlansSection.modals}

        {miscSection.modals}
        {offlinePlansSection.modals}
        {editSection.modals}
      </div>
    )
  },
)

const _LoginRequiredModal = React.memo(
  ({
    state,
    onHide,
  }: {
    state: { heading: string; message: string } | null
    onHide: () => void
  }) => {
    const [lastNonNullState, setLastNonNullState] = useState(state)
    useEffect(() => {
      if (!state) return
      setLastNonNullState(state)
    }, [state])
    const urlUpdater = useURLUpdater()
    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={state !== null}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading">{lastNonNullState?.heading ?? ''}</h2>
        <div className=" dialog-content-div">
          <p className="p-base">{lastNonNullState?.message ?? ''}</p>
        </div>
        <div className=" dialog-button-div">
          <button className=" dialog-button-cancel" onClick={onHide}>
            Cancel
          </button>
          <button
            className=" dialog-button-dark"
            onClick={() => urlUpdater.push(appPaths.login(appPaths.plan()))}
          >
            Login / Sign Up
          </button>
        </div>
      </CenteredModal>
    )
  },
)
