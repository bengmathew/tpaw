import {
  faArrowUp,
  faCalendar,
  faCaretDown,
  faEraser,
  faGrid2,
  faHome,
  faInfinity,
  faPlus,
  faSave,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import Link from 'next/link'
import React, { useMemo, useState } from 'react'
import { appPaths } from '../../../../AppPaths'
import { useUser } from '../../../App/WithUser'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import { setPlansOnDoneURL } from '../../../Plans/Plans'
import {
  SimulationInfoForLinkSrc,
  SimulationInfoForPlanMode,
  useSimulation,
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
import { PlanParamsHelperFns } from '../../../../UseSimulator/PlanParamsHelperFns'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import { PlanMenuActionModalConvertDatingLocal } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingLocal'

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
    const { planParamsNorm } = useSimulation()
    const { datingInfo } = planParamsNorm

    const [loginModalState, setLoginModalState] = useState<{
      heading: string
      message: string
    } | null>(null)
    const [showSwitchDatingModal, setShowSwitchDatingModal] = useState(false)
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

    const convertToDatedElement = (
      <Menu.Item
        as="button"
        className="context-menu-item"
        onClick={() => setShowSwitchDatingModal(true)}
      >
        <span className="context-menu-icon ">
          <FontAwesomeIcon
            icon={datingInfo.isDated ? faInfinity : faCalendar}
          />
        </span>{' '}
        Convert to {datingInfo.isDated ? 'Dateless' : 'Dated'} Plan
      </Menu.Item>
    )

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
                  className="px-3 py-1.5 rounded-lg relative flex  items-center"
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
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
                </Menu.Button>
              )}
              <Menu.Items className="flex flex-col py-2.5 rounded-lg min-w-[250px] max-w-[400px]">
                {user ? (
                  // ---- WITH USER
                  <>
                    <Menu.Item
                      as="button"
                      className=" context-menu-item "
                      onClick={() => setShowSavePlanToAccount(true)}
                    >
                      <span className="context-menu-icon">
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
                      <span className="context-menu-icon">
                        <FontAwesomeIcon icon={faPlus} />
                      </span>{' '}
                      Create a New Plan
                    </Menu.Item>
                    {convertToDatedElement}
                    <Menu.Item>
                      <Link
                        className={'context-menu-item '}
                        href={appPaths.plan()}
                      >
                        <span className="context-menu-icon ">
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
                        <span className="context-menu-icon">
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
                      <span className="context-menu-icon">
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
                      <span className="context-menu-icon">
                        <FontAwesomeIcon icon={faPlus} />
                      </span>{' '}
                      Create a New Plan
                    </Menu.Item>
                    {convertToDatedElement}
                    <Menu.Item>
                      <Link
                        className={'context-menu-item '}
                        href={appPaths.guest()}
                      >
                        <span className="context-menu-icon ">
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
                      <span className="context-menu-icon">
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
                      <span className="context-menu-icon">
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
                      <span className="context-menu-icon">
                        <FontAwesomeIcon icon={faSave} />
                      </span>{' '}
                      Save Plan to Account
                    </Menu.Item>
                    {convertToDatedElement}
                    <Menu.Item>
                      <Link
                        className={'context-menu-item '}
                        href={appPaths.plan()}
                      >
                        <span className="context-menu-icon ">
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
                    className={clix('context-menu-item text-errorFG')}
                    onClick={() => setShowResetModal(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faEraser} />
                    </span>{' '}
                    Undo Changes
                  </Menu.Item>
                )}
              </Menu.Items>
            </ContextModal>
          )}
        </Menu>
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
          reset={() => reset(null)}
        />
        <PlanMenuActionModalConvertDatingLocal
          show={showSwitchDatingModal}
          onHide={() => setShowSwitchDatingModal(false)}
          onConvert={(planParams) => reset(planParams)}
          skipNoUndoCopy
        />
      </div>
    )
  },
)
