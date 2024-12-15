import { faLink } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { useState } from 'react'
import {
  SimulationInfoForServerSrc,
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalCopyToLink } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyToLink'
import { PlanMenuActionViewPlanHistory } from '../PlanMenuActions/PlanMenuActionViewPlanHistory'

export const usePlanMenuSectionMisc = ({
  simulationInfoForServerSrc,
}: {
  simulationInfoForServerSrc: SimulationInfoForServerSrc | null
}) => {
  const { simulationResult } = useSimulationResultInfo()

  const [showCopyToLink, setShowCopyToLink] = useState(false)

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Misc</h2>
      <Menu.Item
        as="button"
        className={'context-menu-item'}
        onClick={() => setShowCopyToLink(true)}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon className="" icon={faLink} />
        </span>{' '}
        Copy to Link
      </Menu.Item>

      {simulationInfoForServerSrc &&
        simulationResult.planParamsNormOfResult.datingInfo.isDated && (
          <PlanMenuActionViewPlanHistory
            className="context-menu-item"
            simulationInfoForServerSrc={simulationInfoForServerSrc}
            nowAsTimestamp={
              simulationResult.planParamsNormOfResult.datingInfo.nowAsTimestamp
            }
          />
        )}
    </div>
  )

  const modals = simulationResult && (
    <>
      <PlanMenuActionModalCopyToLink
        show={showCopyToLink}
        onDone={() => setShowCopyToLink(false)}
        suggestDateless="auto"
      />
    </>
  )
  return { menuItems, modals }
}
