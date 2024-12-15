import { faCopy, faGrid2, faPlus } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { useState } from 'react'
import { appPaths } from '../../../../../AppPaths'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { useURLUpdater } from '../../../../../Utils/UseURLUpdater'
import { PlanLocalStorage } from '../../../PlanRootLocalMain/PlanLocalStorage'
import { useSimulationResultInfo } from '../../../PlanRootHelpers/WithSimulation'

export const usePlanMenuSectionManagePlansNotLoggedIn = () => {
  const urlUpdater = useURLUpdater()
  const { simulationResult } = useSimulationResultInfo()
  const [hasLocal] = useState(() => PlanLocalStorage.read() !== null)
  const [showLoginNeededModal, setShowLoginNeededModal] = useState(false)

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Manage Plans</h2>
      <Menu.Item
        as="button"
        className="context-menu-item"
        onClick={() => {
          hasLocal
            ? setShowLoginNeededModal(true)
            : urlUpdater.push(appPaths.plan())
        }}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon icon={faPlus} />
        </span>{' '}
        New
      </Menu.Item>
      <Menu.Item
        as="button"
        className="context-menu-item"
        onClick={() => setShowLoginNeededModal(true)}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon icon={faCopy} />
        </span>{' '}
        Copy
      </Menu.Item>

      <Menu.Item
        as="button"
        className="context-menu-item"
        onClick={() => setShowLoginNeededModal(true)}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon icon={faGrid2} />
        </span>{' '}
        View All
      </Menu.Item>
    </div>
  )

  const modals = (
    <>
      <CenteredModal
        className=" dialog-outer-div"
        show={showLoginNeededModal}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading">Login to Continue</h2>
        <div className=" dialog-content-div">
          <p className="p-base">You need to login to perform this action.</p>
        </div>
        <div className=" dialog-button-div">
          <button
            className=" dialog-button-cancel"
            onClick={() => setShowLoginNeededModal(false)}
          >
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
    </>
  )
  return { menuItems, modals }
}
