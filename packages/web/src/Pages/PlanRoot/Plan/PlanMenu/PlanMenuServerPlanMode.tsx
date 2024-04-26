import {
  faArrowUp,
  faCalendar,
  faCaretDown,
  faCopy,
  faEraser,
  faGrid2,
  faHome,
  faInfinity,
  faPlus,
  faSlash,
  faTag,
  faTimes,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import Link from 'next/link'
import React, { useMemo, useState } from 'react'
import { appPaths } from '../../../../AppPaths'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import { setPlansOnDoneURL } from '../../../Plans/Plans'
import {
  SimulationInfoForPlanMode,
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionCopyToLink } from './PlanMenuActions/PlanMenuActionCopyToLink'
import { PlanMenuActionModalCopyServer } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyServer'
import { PlanMenuActionModalCreatePlan } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCreatePlan'
import { PlanMenuActionModalDelete } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalDelete'
import { PlanMenuActionModalEditLabel } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalEditLabel'
import { PlanMenuActionModalReset } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalReset'
import { PlanMenuActionModalSetAsMain } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSetAsMain'
import { PlanMenuActionViewPlanHistory } from './PlanMenuActions/PlanMenuActionViewPlanHistory'
import { PlanMenuDivider } from './PlanMenuHelpers/PlanMenuDivider'
import { PlanMenuSubMenuUndoRedo } from './PlanMenuSubMenu/PlanMenuSubMenuUndoRedo'
import { PlanParamsHelperFns } from '../../../../UseSimulator/PlanParamsHelperFns'
import { CurrentPortfolioBalance } from '../../PlanRootHelpers/CurrentPortfolioBalance'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import { faBan } from '@fortawesome/pro-light-svg-icons'
import { PlanMenuActionModalConvertDatingServer } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingServer'

export const PlanMenuServerPlanMode = React.memo(
  ({
    simulationInfoForServerSrc,
    simulationInfoForPlanMode,
  }: {
    simulationInfoForServerSrc: SimulationInfoForServerSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const { ianaTimezoneName } = useIANATimezoneName()
    const planColors = usePlanColors()
    const { planPaths, planParamsNorm, currentPortfolioBalanceInfo } =
      useSimulation()
    const { datingInfo } = planParamsNorm
    const { plan, syncState, reload } = simulationInfoForServerSrc
    const isSyncing = syncState.type !== 'synced'
    const label = plan.isMain ? 'Main Plan' : plan.label ?? 'Untitled'

    const currentPortfolioBalanceAmount =
      currentPortfolioBalanceInfo.isDatedPlan
        ? CurrentPortfolioBalance.getAmountInfo(
            currentPortfolioBalanceInfo.info,
          ).amount
        : currentPortfolioBalanceInfo.amount
    const planParamsForDatingSwitch = useMemo(
      () =>
        planParamsNorm.datingInfo.isDated
          ? PlanParamsHelperFns.toDatelessPlan(
              planParamsNorm,
              currentPortfolioBalanceAmount,
            )
          : PlanParamsHelperFns.toDatedPlan(planParamsNorm, ianaTimezoneName),
      [currentPortfolioBalanceAmount, ianaTimezoneName, planParamsNorm],
    )

    const [showCreatePlanModal, setShowCreatePlanModal] = useState(false)
    const [showEditLabelModal, setShowEditLabelModal] = useState(false)
    const [showSetAsMainModal, setShowSetAsMainModal] = React.useState(false)
    const [showCopyModal, setShowCopyModal] = useState(false)
    const [showConvertDatingModal, setShowConvertDatingModal] = useState(false)
    const [showResetModal, setShowResetModal] = useState(false)
    const [showDeleteModal, setShowDeleteModal] = useState(false)

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
                  className="px-3 py-1.5 rounded-lg"
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
              <Menu.Items className="flex flex-col py-2.5 rounded-lg min-w-[275px]">
                <Menu.Item
                  as="button"
                  className="context-menu-item"
                  onClick={() => setShowCreatePlanModal(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faPlus} />
                  </span>{' '}
                  Create a New Plan
                </Menu.Item>
                <Menu.Item
                  as="button"
                  className="context-menu-item"
                  onClick={() => setShowCopyModal(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faCopy} />
                  </span>{' '}
                  Copy to New Plan
                </Menu.Item>

                {!plan.isMain && (
                  <Menu.Item
                    as="button"
                    className=" context-menu-item "
                    onClick={() => setShowSetAsMainModal(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faArrowUp} />
                    </span>{' '}
                    Make This the Main Plan
                  </Menu.Item>
                )}
                {plan.isDated ? (
                  <Menu.Item
                    as="button"
                    className="context-menu-item"
                    onClick={() => setShowConvertDatingModal(true)}
                  >
                    <span className="context-menu-icon ">
                      <FontAwesomeIcon icon={faInfinity} />
                    </span>{' '}
                    Convert to Dateless Plan
                  </Menu.Item>
                ) : (
                  <Menu.Item
                    as="button"
                    className="context-menu-item"
                    onClick={() => setShowConvertDatingModal(true)}
                  >
                    <span className="context-menu-icon ">
                      <FontAwesomeIcon icon={faCalendar} />
                    </span>{' '}
                    Convert to Dated Plan
                  </Menu.Item>
                )}
                {!plan.isMain && (
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
                )}

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
                <PlanMenuDivider />

                {!plan.isMain && (
                  <Menu.Item
                    as="button"
                    className="context-menu-item"
                    onClick={() => setShowEditLabelModal(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faTag} />
                    </span>{' '}
                    Edit Label
                  </Menu.Item>
                )}

                <PlanMenuActionCopyToLink
                  className="context-menu-item"
                  closeMenu={close}
                />
                {datingInfo.isDated && (
                  <PlanMenuActionViewPlanHistory
                    className="context-menu-item"
                    simulationInfoForServerSrc={simulationInfoForServerSrc}
                    nowAsTimestamp={datingInfo.nowAsTimestamp}
                  />
                )}
                <Menu.Item
                  as="button"
                  className="context-menu-item text-errorFG"
                  onClick={() => setShowResetModal(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faEraser} />
                  </span>{' '}
                  Reset
                </Menu.Item>
                {!plan.isMain && (
                  <Menu.Item
                    as="button"
                    className="context-menu-item text-errorFG"
                    onClick={() => setShowDeleteModal(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faTrash} />
                    </span>{' '}
                    Delete
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
        <PlanMenuActionModalCreatePlan
          show={showCreatePlanModal}
          onHide={() => setShowCreatePlanModal(false)}
          switchOnCreate={{ onBeforeNav: null }}
        />
        <PlanMenuActionModalEditLabel
          show={showEditLabelModal}
          plan={plan}
          onHide={() => setShowEditLabelModal(false)}
        />
        <PlanMenuActionModalSetAsMain
          show={showSetAsMainModal}
          plan={plan}
          isSyncing={isSyncing}
          onHide={() => setShowSetAsMainModal(false)}
          switchToMainPlanOnSuccess
        />
        <PlanMenuActionModalCopyServer
          show={showCopyModal}
          plan={plan}
          onHide={() => setShowCopyModal(false)}
          isSyncing={isSyncing}
          hideOnSuccess={false}
          cutAfterId={null}
          planParamsForDatingSwitch={planParamsForDatingSwitch}
        />
        <PlanMenuActionModalConvertDatingServer
          show={showConvertDatingModal}
          onHide={() => setShowConvertDatingModal(false)}
          plan={plan}
          isSyncing={isSyncing}
          reload={reload}
        />
        <PlanMenuActionModalDelete
          show={showDeleteModal}
          plan={plan}
          onHide={() => setShowDeleteModal(false)}
          toPlansOnSuccess={true}
          isSyncing={isSyncing}
        />
        <PlanMenuActionModalReset
          show={showResetModal}
          isSyncing={isSyncing}
          plan={plan}
          onHide={() => setShowResetModal(false)}
          reloadOnSuccess={{ planPaths, reload }}
        />
      </div>
    )
  },
)
