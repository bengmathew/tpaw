import {
  faArrowUp,
  faCopy,
  faGrid2,
  faHome,
  faPlus,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import Link from 'next/link'
import { useState } from 'react'
import { appPaths } from '../../../../../AppPaths'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { setPlansOnDoneURL } from '../../../../Plans/Plans'
import { SimulationInfoForServerSrc } from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalCopyServer } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyServer'
import { PlanMenuActionModalCreatePlan } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCreatePlan'
import { PlanMenuActionModalSetAsMain } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSetAsMain'

export const usePlanMenuSectionManagePlansLoggedIn = ({
  simulationInfoForServerSrc,
  isModified,
}: {
  simulationInfoForServerSrc: SimulationInfoForServerSrc | null
  isModified: false | { setForceNav: () => void }
}) => {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [showSetAsMainModal, setShowSetAsMainModal] = useState(false)
  const [showWarnModifiedAndCreateModal, setShowModifiedAndCreateModal] =
    useState(false)

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Manage Plans</h2>
      <Menu.Item
        as="button"
        className="context-menu-item"
        onClick={() => {
          isModified
            ? setShowModifiedAndCreateModal(true)
            : setShowCreateModal(true)
        }}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon icon={faPlus} />
        </span>{' '}
        New
      </Menu.Item>
      {simulationInfoForServerSrc && (
        <Menu.Item
          as="button"
          className="context-menu-item"
          onClick={() => setShowDuplicateModal(true)}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faCopy} />
          </span>{' '}
          Copy
        </Menu.Item>
      )}

      {simulationInfoForServerSrc &&
        !simulationInfoForServerSrc.plan.isMain && (
          <Menu.Item
            as="button"
            className=" context-menu-item "
            onClick={() => setShowSetAsMainModal(true)}
          >
            <span className="context-menu-icon">
              <FontAwesomeIcon icon={faArrowUp} />
            </span>{' '}
            Set as Main
          </Menu.Item>
        )}
      {(!simulationInfoForServerSrc ||
        !simulationInfoForServerSrc.plan.isMain) && (
        <Menu.Item>
          <Link className={'context-menu-item '} href={appPaths.plan()}>
            <span className="context-menu-icon ">
              <FontAwesomeIcon className="" icon={faHome} />
            </span>{' '}
            Open Main 
          </Link>
        </Menu.Item>
      )}

      <Menu.Item>
        <Link
          className={'context-menu-item'}
          href={appPaths.plans()}
          onClick={() => setPlansOnDoneURL()}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faGrid2} />
          </span>{' '}
          View All
        </Link>
      </Menu.Item>
    </div>
  )

  const modals = (
    <>
      <PlanMenuActionModalCreatePlan
        show={showCreateModal}
        onHide={() => setShowCreateModal(false)}
        switchOnCreate={{
          onBeforeNav: isModified ? isModified.setForceNav : null,
        }}
      />
      {simulationInfoForServerSrc && (
        <PlanMenuActionModalCopyServer
          show={showDuplicateModal}
          plan={simulationInfoForServerSrc.plan}
          onHide={() => setShowDuplicateModal(false)}
          isSyncing={simulationInfoForServerSrc.syncState.type !== 'synced'}
          hideOnSuccess={false}
          cutAfterId={null}
        />
      )}

      {simulationInfoForServerSrc && (
        <PlanMenuActionModalSetAsMain
          show={showSetAsMainModal}
          plan={simulationInfoForServerSrc.plan}
          isSyncing={simulationInfoForServerSrc.syncState.type !== 'synced'}
          onHide={() => setShowSetAsMainModal(false)}
          switchToMainPlanOnSuccess
        />
      )}
      <CenteredModal
        className=" dialog-outer-div"
        show={showWarnModifiedAndCreateModal}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading"></h2>
        <div className=" dialog-content-div">
          <p className=" p-base">
            You have unsaved changes. Save your plan before creating a new one
            if you {"don't"} want to lose these changes.
          </p>
        </div>
        <div className=" dialog-button-div">
          <button
            className=" dialog-button-cancel"
            onClick={() => setShowModifiedAndCreateModal(false)}
          >
            Cancel
          </button>
          <button
            className=" dialog-button-warning"
            onClick={() => {
              setShowModifiedAndCreateModal(false)
              setShowCreateModal(true)
            }}
          >
            Create Anyway
          </button>
        </div>
      </CenteredModal>
    </>
  )
  return { menuItems, modals }
}
