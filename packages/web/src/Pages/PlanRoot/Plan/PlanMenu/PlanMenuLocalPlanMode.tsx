import {
  faCaretDown,
  faCopy,
  faEraser,
  faGrid2,
  faPlus,
  faUser,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import React, { useState } from 'react'
import { appPaths } from '../../../../AppPaths'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { ContextMenu2 } from '../../../Common/Modal/ContextMenu2'
import {
  SimulationInfoForLocalMainSrc,
  SimulationInfoForPlanMode,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionCopyToLink } from './PlanMenuActions/PlanMenuActionCopyToLink'
import { PlanMenuActionModalLoginRequired } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLoginRequired'
import { PlanMenuActionModalResetLocal } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalResetLocal'
import { PlanMenuActionPrint } from './PlanMenuActions/PlanMenuActionPrint'
import { PlanMenuDivider } from './PlanMenuHelpers/PlanMenuDivider'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'
import Link from 'next/link'

export const PlanMenuLocalPlanMode = React.memo(
  ({
    simulationInfoForLocalMainSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForLocalMainSrc: SimulationInfoForLocalMainSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()
    const { reset } = simulationInfoForLocalMainSrc

    const urlUpdater = useURLUpdater()
    const [loginModalState, setLoginModalState] = useState<{
      heading: string
      message: string
    } | null>(null)
    const [showResetModal, setShowResetModal] = useState(false)

    return (
      <div className="flex gap-x-2">
        <ContextMenu2
          className="px-3 py-1.5 rounded-lg"
          align="right"
          style={{
            backgroundColor: planColors.results.bg,
            color: planColors.results.fg,
          }}
        >
          <div className="relative ">
            <h2 className="text-start font-semibold">Guest Plan</h2>
            <div className="flex items-center">
              <h2 className="mt-1 max-w-[200px] whitespace-nowrap text-ellipsis overflow-hidden ">
                Saved on Browser
              </h2>
              <FontAwesomeIcon className=" -mb-1 ml-2" icon={faCaretDown} />
            </div>
          </div>
          {({ close, onMenuClose }) => (
            <Menu.Items className="flex flex-col py-2.5 rounded-lg min-w-[250px] max-w-[400px]">
              <Menu.Item>
                <Link
                  className={'context-menu-item '}
                  href={appPaths.login(appPaths.plan())}
                >
                  <span className="inline-block w-[25px] ">
                    <FontAwesomeIcon className="" icon={faUser} />
                  </span>{' '}
                  Save Plan to Account
                </Link>
              </Menu.Item>
              <Menu.Item
                as="button"
                className="context-menu-item "
                onClick={() =>
                  setLoginModalState({
                    heading: 'Login for Multiple Plans',
                    message:
                      'You need to be logged in to have more than one plan.',
                  })
                }
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faPlus} />
                </span>{' '}
                Create a New Plan
              </Menu.Item>
              <Menu.Item
                as="button"
                className="context-menu-item "
                onClick={() =>
                  setLoginModalState({
                    heading: 'Login for Multiple Plans',
                    message:
                      'You need to be logged in to have more than one plan.',
                  })
                }
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faCopy} />
                </span>{' '}
                Copy to New Plan
              </Menu.Item>
              <Menu.Item
                as="button"
                className="context-menu-item "
                onClick={() =>
                  setLoginModalState({
                    heading: 'Login for Multiple Plans',
                    message:
                      'You need to be logged in to have more than one plan.',
                  })
                }
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faGrid2} />
                </span>{' '}
                View All Plans
              </Menu.Item>
              <PlanMenuDivider />
              <PlanMenuActionCopyToLink
                className="context-menu-item "
                closeMenu={close}
              />

              <PlanMenuActionPrint
                className="context-menu-item"
                closeMenu={close}
              />
              <Menu.Item
                as="button"
                className="context-menu-item "
                onClick={() =>
                  setLoginModalState({
                    heading: 'Login to Track Plan History',
                    message:
                      'History is availabe only for plans saved to your account. Login to start tracking history for this plan.',
                  })
                }
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faCopy} />
                </span>{' '}
                View Plan History
              </Menu.Item>
              <Menu.Item
                as="button"
                className="context-menu-item text-errorFG"
                onClick={() => setShowResetModal(true)}
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faEraser} />
                </span>{' '}
                Reset
              </Menu.Item>
            </Menu.Items>
          )}
        </ContextMenu2>
        <PlanMenuSubMenuUndoRedo
          simulationDetailForPlanMode={simulationInfoForPlanMode}
          className={{ undo: 'pl-6 pr-3', redo: 'pl-3 pr-6' }}
        />

        <PlanMenuActionModalLoginRequired
          state={loginModalState}
          onHide={() => setLoginModalState(null)}
        />
        <PlanMenuActionModalResetLocal
          show={showResetModal}
          onHide={() => setShowResetModal(false)}
          title="Reset"
          message="Are you sure you want to reset this plan? This cannot be undone."
          reset={reset}
        />
      </div>
    )
  },
)
