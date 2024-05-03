import { faEraser } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { useState } from 'react'
import {
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalResetServer } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalResetServer'
import { PlanMenuActionModalConvertDatingServer } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingServer'
import {
  faCalendar,
  faInfinity,
  faTag,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { PlanMenuActionModalDelete } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalDelete'
import { PlanMenuActionModalEditLabel } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalEditLabel'
import { convertToDatelessLabel } from './PlanMenuSectionEditLocal'

export const usePlanMenuSectionEditServer = ({
  simulationInfoForServerSrc,
}: {
  simulationInfoForServerSrc: SimulationInfoForServerSrc
}) => {
  const { planPaths } = useSimulation()

  const [showEditLabelModal, setShowEditLabelModal] = useState(false)
  const [showConvertDatingModal, setShowConvertDatingModal] = useState(false)
  const [showResetModal, setShowResetModal] = useState(false)
  const [showDeleteModal, setShowDeleteModal] = useState(false)

  const { syncState, plan, reload } = simulationInfoForServerSrc
  const isSyncing = syncState.type !== 'synced'

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Modify Plan</h2>
      {!plan.isMain && (
        <Menu.Item
          as="button"
          className="context-menu-item-indent"
          onClick={() => setShowEditLabelModal(true)}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faTag} />
          </span>{' '}
          Edit Label
        </Menu.Item>
      )}
      <Menu.Item
        as="button"
        className="context-menu-item-indent"
        onClick={() => setShowConvertDatingModal(true)}
      >
        <span className="context-menu-icon ">
          <FontAwesomeIcon icon={plan.isDated ? faInfinity : faCalendar} />
        </span>{' '}
        {plan.isDated ? convertToDatelessLabel : 'Convert to  Dated'}
      </Menu.Item>
      <Menu.Item
        as="button"
        className="context-menu-item-indent text-errorFG"
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
          className="context-menu-item-indent text-errorFG"
          onClick={() => setShowDeleteModal(true)}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faTrash} />
          </span>{' '}
          Delete
        </Menu.Item>
      )}
    </div>
  )

  const modals = (
    <>
      <PlanMenuActionModalEditLabel
        show={showEditLabelModal}
        plan={plan}
        onHide={() => setShowEditLabelModal(false)}
      />
      <PlanMenuActionModalConvertDatingServer
        show={showConvertDatingModal}
        onHide={() => setShowConvertDatingModal(false)}
        plan={plan}
        isSyncing={isSyncing}
        reload={reload}
      />
      <PlanMenuActionModalResetServer
        show={showResetModal}
        isSyncing={isSyncing}
        plan={plan}
        onHide={() => setShowResetModal(false)}
        reloadOnSuccess={{ planPaths, reload }}
      />
      <PlanMenuActionModalDelete
        show={showDeleteModal}
        plan={plan}
        onHide={() => setShowDeleteModal(false)}
        toPlansOnSuccess={true}
        isSyncing={isSyncing}
      />
    </>
  )

  return { menuItems, modals }
}
