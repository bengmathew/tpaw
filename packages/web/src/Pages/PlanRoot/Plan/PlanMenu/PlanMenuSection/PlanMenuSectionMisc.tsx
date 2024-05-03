import { faLink } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { useState } from 'react'
import {
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalCopyToLink } from '../PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyToLink'
import { PlanMenuActionViewPlanHistory } from '../PlanMenuActions/PlanMenuActionViewPlanHistory'

export const usePlanMenuSectionMisc = ({
  simulationInfoForServerSrc,
}: {
  simulationInfoForServerSrc: SimulationInfoForServerSrc | null
}) => {
  const { planParamsNorm } = useSimulation()
  const { datingInfo } = planParamsNorm

  const [showCopyToLink, setShowCopyToLink] = useState(false)

  const menuItems = (
    <div className="context-menu-section">
      <h2 className=" context-menu-section-heading">Misc</h2>
      <Menu.Item
        as="button"
        className={'context-menu-item-indent'}
        onClick={() => setShowCopyToLink(true)}
      >
        <span className="context-menu-icon">
          <FontAwesomeIcon className="" icon={faLink} />
        </span>{' '}
        Copy to Link
      </Menu.Item>

      {simulationInfoForServerSrc && datingInfo.isDated && (
        <PlanMenuActionViewPlanHistory
          className="context-menu-item-indent"
          simulationInfoForServerSrc={simulationInfoForServerSrc}
          nowAsTimestamp={datingInfo.nowAsTimestamp}
        />
      )}
    </div>
  )

  const modals = (
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
