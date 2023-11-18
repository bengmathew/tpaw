import {
  faArrowUp,
  faCaretDown,
  faEraser,
  faGrid2,
  faHome,
  faPlus,
  faSave,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clsx from 'clsx'
import Link from 'next/link'
import React, { useState } from 'react'
import { appPaths } from '../../../../AppPaths'
import { useUser } from '../../../App/WithUser'
import { ContextMenu2 } from '../../../Common/Modal/ContextMenu2'
import { setPlansOnDoneURL } from '../../../Plans/Plans'
import {
  SimulationInfoForLinkSrc,
  SimulationInfoForPlanMode,
} from '../../PlanRootHelpers/WithSimulation'
import { PlanRootLinkUnsavedWarningAlert } from '../../PlanRootLink/PlanRootLinkUnsavedWarningAlert'
import { PlanLocalStorage } from '../../PlanRootLocalMain/PlanLocalStorage'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionCopyToLink } from './PlanMenuActions/PlanMenuActionCopyToLink'
import { PlanMenuActionModalCreatePlan } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCreatePlan'
import { PlanMenuActionModalLoginAndSaveLink } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLoginAndSaveLink'
import { PlanMenuActionModalLoginRequired } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLoginRequired'
import { PlanMenuActionModalOverwriteLocal } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalOverwriteLocal'
import { PlanMenuActionModalResetLocal } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalResetLocal'
import { PlanMenuActionModalSaveLinkPlanToAccount } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSaveLinkPlanToAccount'
import { PlanMenuDivider } from './PlanMenuHelpers/PlanMenuDivider'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'

export const PlanMenuLinkPlanMode = React.memo(
  ({
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()
    const user = useUser()

    const [loginModalState, setLoginModalState] = useState<{
      heading: string
      message: string
    } | null>(null)
    const [showOverwriteLocalModal, setShowOverwriteLocalModal] =
      useState(false)
    const [showModifiedAndCreateModal, setShowModifiedAndCreateModal] =
      useState(false)
    const [showCreatePlanModal, setShowCreatePlanModal] = useState(false)

    const [hasLocal] = useState(() => PlanLocalStorage.read() !== null)

    const [showResetModal, setShowResetModal] = useState(false)
    const [showSavePlanToAccount, setShowSavePlanToAccount] = useState(false)
    const [showLoginAndSavePlan, setShowLoginAndSavePlan] = useState(false)
    const { planParamsUndoRedoStack } = simulationInfoForPlanMode
    const { isModified, reset, setForceNav } = simulationInfoForLinkSrc

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
          <div className="relative flex  items-center">
            <h2 className="text-start font-semibold">
              Plan From Link{' '}
              {isModified && (
                <span className="">
                  {' '}
                  - <span className="text-errorFG"> Modified</span>
                </span>
              )}{' '}
            </h2>
            <FontAwesomeIcon className="ml-2" icon={faCaretDown} />
          </div>
          {({ close }) => (
            <Menu.Items className="flex flex-col py-2.5 rounded-lg min-w-[250px] max-w-[400px]">
              {user ? (
                // ---- WITH USER
                <>
                  <Menu.Item
                    as="button"
                    className=" context-menu-item "
                    onClick={() => setShowSavePlanToAccount(true)}
                  >
                    <span className="inline-block w-[25px]">
                      <FontAwesomeIcon icon={faSave} />
                    </span>{' '}
                    Save Plan to Account
                  </Menu.Item>
                  <Menu.Item
                    as="button"
                    className="context-menu-item"
                    onClick={() => {
                      if (isModified) {
                        setShowModifiedAndCreateModal(true)
                      } else {
                        setShowCreatePlanModal(true)
                      }
                    }}
                  >
                    <span className="inline-block w-[25px]">
                      <FontAwesomeIcon icon={faPlus} />
                    </span>{' '}
                    Create a New Plan
                  </Menu.Item>
                  <Menu.Item>
                    <Link
                      className={'context-menu-item '}
                      href={appPaths.plan()}
                    >
                      <span className="inline-block w-[25px] ">
                        <FontAwesomeIcon className="" icon={faHome} />
                      </span>{' '}
                      Switch to Main Plan
                    </Link>
                  </Menu.Item>
                  <Menu.Item>
                    <Link
                      className={'context-menu-item '}
                      href={appPaths.plans()}
                      onClick={() => setPlansOnDoneURL()}
                    >
                      <span className="inline-block w-[25px]">
                        <FontAwesomeIcon icon={faGrid2} />
                      </span>{' '}
                      View All Plans
                    </Link>
                  </Menu.Item>
                </>
              ) : hasLocal ? (
                // ---- NO USER, HAS LOCAL
                <>
                  <Menu.Item
                    as="button"
                    className=" context-menu-item "
                    onClick={() => setShowLoginAndSavePlan(true)}
                  >
                    <span className="inline-block w-[25px]">
                      <FontAwesomeIcon icon={faSave} />
                    </span>{' '}
                    Save Plan to Account
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
                  <Menu.Item>
                    <Link
                      className={'context-menu-item '}
                      href={appPaths.guest()}
                    >
                      <span className="inline-block w-[25px] ">
                        <FontAwesomeIcon className="" icon={faHome} />
                      </span>{' '}
                      Switch to Your Plan
                    </Link>
                  </Menu.Item>
                  <Menu.Item
                    as="button"
                    className=" context-menu-item "
                    onClick={() => setShowOverwriteLocalModal(true)}
                  >
                    <span className="inline-block w-[25px]">
                      <FontAwesomeIcon icon={faArrowUp} />
                    </span>{' '}
                    Make This Your Plan
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
                </>
              ) : (
                // ---- NO USER, NO LOCAL
                <>
                  <Menu.Item
                    as="button"
                    className=" context-menu-item "
                    onClick={() => setShowLoginAndSavePlan(true)}
                  >
                    <span className="inline-block w-[25px]">
                      <FontAwesomeIcon icon={faSave} />
                    </span>{' '}
                    Save Plan to Account
                  </Menu.Item>
                  <Menu.Item>
                    <Link
                      className={'context-menu-item '}
                      href={appPaths.plan()}
                    >
                      <span className="inline-block w-[25px] ">
                        <FontAwesomeIcon className="" icon={faPlus} />
                      </span>{' '}
                      Create Your Own Plan
                    </Link>
                  </Menu.Item>
                </>
              )}
              <PlanMenuDivider />

              <PlanMenuActionCopyToLink
                className=" context-menu-item "
                closeMenu={close}
              />
              {isModified && (
                <Menu.Item
                  as="button"
                  className={clsx('context-menu-item text-errorFG')}
                  onClick={() => setShowResetModal(true)}
                >
                  <span className="inline-block w-[25px]">
                    <FontAwesomeIcon icon={faEraser} />
                  </span>{' '}
                  Undo Changes
                </Menu.Item>
              )}
            </Menu.Items>
          )}
        </ContextMenu2>
        <PlanMenuSubMenuUndoRedo
          simulationDetailForPlanMode={simulationInfoForPlanMode}
          className={{ undo: 'pl-6 pr-3', redo: 'pl-3 pr-6' }}
        />
        <PlanRootLinkUnsavedWarningAlert
          show={showModifiedAndCreateModal}
          onCancel={() => setShowModifiedAndCreateModal(false)}
          onLeave={() => {
            setShowModifiedAndCreateModal(false)
            setShowCreatePlanModal(true)
          }}
        />
        <PlanMenuActionModalCreatePlan
          show={showCreatePlanModal}
          onHide={() => setShowCreatePlanModal(false)}
          switchOnCreate={{ onBeforeNav: setForceNav }}
        />
        <PlanMenuActionModalLoginRequired
          state={loginModalState}
          onHide={() => setLoginModalState(null)}
        />
        <PlanMenuActionModalOverwriteLocal
          show={showOverwriteLocalModal}
          onHide={() => setShowOverwriteLocalModal(false)}
          planPramsUndoRedoStack={planParamsUndoRedoStack}
          setForceNav={setForceNav}
        />
        <PlanMenuActionModalSaveLinkPlanToAccount
          show={showSavePlanToAccount}
          onHide={() => setShowSavePlanToAccount(false)}
          simulationInfoForLinkSrc={simulationInfoForLinkSrc}
          simulationInfoForPlanMode={simulationInfoForPlanMode}
        />
        <PlanMenuActionModalLoginAndSaveLink
          show={showLoginAndSavePlan}
          onHide={() => setShowLoginAndSavePlan(false)}
          simulationInfoForLinkSrc={simulationInfoForLinkSrc}
          simulationInfoForPlanMode={simulationInfoForPlanMode}
        />
        <PlanMenuActionModalResetLocal
          show={showResetModal}
          onHide={() => setShowResetModal(false)}
          title="Undo Changes"
          message="Are you sure you want to undo the changes you made? This will reset the plan to the original version shared through the link."
          reset={reset}
        />
      </div>
    )
  },
)
