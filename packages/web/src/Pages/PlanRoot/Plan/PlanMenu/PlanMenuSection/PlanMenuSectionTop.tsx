import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { useState } from 'react'
import {
  SimulationInfoForFileSrc,
  SimulationInfoForLinkSrc,
  SimulationInfoForPlanMode,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanLocalStorage } from '../../../PlanRootLocalMain/PlanLocalStorage'
import { PlanMenuActionModalCopyToLink } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyToLink'
import { faSave } from '@fortawesome/pro-solid-svg-icons'
import { assert } from '@tpaw/common'
import { useUser } from '../../../../App/WithUser'
import { PlanMenuActionModalLoginAndSaveLink } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalLoginAndSaveWorkingPlan'
import { PlanMenuActionModalOverwriteLocal } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalOverwriteLocal'
import { PlanMenuActionModalSaveWorkingPlanToAccount } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSaveWorkingPlanToAccount'
import { PlanFileDataFns } from '../../../PlanRootFile/PlanFileData'

export const usePlanMenuSectionTop = ({
  simulationInfoBySrc,
  simulationInfoForPlanMode,
}: {
  simulationInfoForPlanMode: SimulationInfoForPlanMode
  simulationInfoBySrc: SimulationInfoForFileSrc | SimulationInfoForLinkSrc
}) => {
  const user = useUser()
  const { planParamsUndoRedoStack } = simulationInfoForPlanMode
  const { setForceNav } = simulationInfoBySrc
  const [hasLocal] = useState(() => PlanLocalStorage.read() !== null)
  const [showSavePlanToAccount, setShowSavePlanToAccount] = useState(false)
  const [showOverwriteLocalModal, setShowOverwriteLocalModal] = useState(false)
  const [showLoginAndSavePlan, setShowLoginAndSavePlan] = useState(false)

  const label =
    simulationInfoBySrc.src === 'link'
      ? null
      : PlanFileDataFns.labelFromFilename(simulationInfoBySrc.plan.filename)

  const menuItems = (
    <>
      {hasLocal ? (
        <Menu.Item
          as="button"
          className=" context-menu-item "
          onClick={() => {
            assert(!user)
            setShowOverwriteLocalModal(true)
          }}
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faSave} />
          </span>{' '}
          Save as Guest Plan
        </Menu.Item>
      ) : (
        <Menu.Item
          as="button"
          className=" context-menu-item "
          onClick={() =>
            user
              ? setShowSavePlanToAccount(true)
              : setShowLoginAndSavePlan(true)
          }
        >
          <span className="context-menu-icon">
            <FontAwesomeIcon icon={faSave} />
          </span>{' '}
          Save Plan to Account
        </Menu.Item>
      )}
    </>
  )

  const modals = (
    <>
      <PlanMenuActionModalOverwriteLocal
        show={showOverwriteLocalModal}
        onHide={() => setShowOverwriteLocalModal(false)}
        planPramsUndoRedoStack={planParamsUndoRedoStack}
        setForceNav={setForceNav}
      />
      <PlanMenuActionModalSaveWorkingPlanToAccount
        show={showSavePlanToAccount}
        onHide={() => setShowSavePlanToAccount(false)}
        setForceNav={setForceNav}
        label={label}
        simulationInfoForPlanMode={simulationInfoForPlanMode}
      />
      <PlanMenuActionModalLoginAndSaveLink
        show={showLoginAndSavePlan}
        onHide={() => setShowLoginAndSavePlan(false)}
        setForceNav={setForceNav}
        label={label}
        simulationInfoForPlanMode={simulationInfoForPlanMode}
      />
    </>
  )
  return { menuItems, modals }
}
