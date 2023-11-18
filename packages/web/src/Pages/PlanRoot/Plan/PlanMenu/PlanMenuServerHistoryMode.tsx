import {
  faArrowRightFromBracket,
  faCaretDown,
  faCopy,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clsx from 'clsx'
import Link from 'next/link'
import React, { useState } from 'react'
import { ContextMenu2 } from '../../../Common/Modal/ContextMenu2'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForHistoryMode,
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionCopyToLink } from './PlanMenuActions/PlanMenuActionCopyToLink'
import { PlanMenuActionModalCopyServer } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyServer'
import { PlanMenuDivider } from './PlanMenuHelpers/PlanMenuDivider'
import { PlanMenuSubMenuRewind } from './PlanMenuSubMenu/PlanMenuSubMenuRewind'

export const PlanMenuServerHistoryMode = React.memo(
  ({
    simulationInfoForServerSrc,
    simulationInfoForHistoryMode,
  }: {
    simulationInfoForServerSrc: SimulationInfoForServerSrc
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
  }) => {
    const planColors = usePlanColors()
    const { getZonedTime } = useIANATimezoneName()
    const { planParamsId, currentTimestamp } = useSimulation()
    const { plan, isSyncing } = simulationInfoForServerSrc
    const [showCopyModal, setShowCopyModal] = useState(false)
    const label = plan.isMain ? 'Main Plan' : plan.label ?? 'Untitled'

    const exitHistoryURL = new URL(window.location.href)
    exitHistoryURL.searchParams.delete('rewindTo')
    return (
      <div className="flex gap-x-2">
        <ContextMenu2
          className={clsx('px-3 py-1.5 rounded-lg')}
          align="right"
          style={{
            backgroundColor: planColors.results.bg,
            color: planColors.results.fg,
          }}
        >
          <div className="">
            <h2 className="font-bold">History Mode</h2>
            <div className="flex items-center justify-between gap-x-2">
              <h2
                className="text-start max-w-[200px] overflow-hidden whitespace-nowrap text-ellipsis"
                title={label}
              >
                {label}
              </h2>
              <FontAwesomeIcon icon={faCaretDown} />
            </div>
          </div>
          {({ close, onMenuClose }) => (
            <Menu.Items className="flex flex-col py-2.5 rounded-lg">
              <Menu.Item>
                <button
                  className="context-menu-item "
                  onClick={() => setShowCopyModal(true)}
                >
                  <span className="inline-block w-[25px]">
                    <FontAwesomeIcon icon={faCopy} />
                  </span>{' '}
                  Copy to New Plan
                </button>
              </Menu.Item>
              <PlanMenuActionCopyToLink
                className="context-menu-item"
                closeMenu={close}
              />
              <PlanMenuDivider />
              <Menu.Item>
                <Link
                  className="context-menu-item "
                  href={exitHistoryURL}
                  shallow
                >
                  <span className="inline-block w-[25px]">
                    <FontAwesomeIcon icon={faArrowRightFromBracket} />
                  </span>{' '}
                  Exit History Mode
                </Link>
              </Menu.Item>
            </Menu.Items>
          )}
        </ContextMenu2>
        <PlanMenuSubMenuRewind
          simulationInfoForHistoryMode={simulationInfoForHistoryMode}
        />

        <PlanMenuActionModalCopyServer
          show={showCopyModal}
          plan={plan}
          onHide={() => setShowCopyModal(false)}
          hideOnSuccess={false}
          cutAfterId={planParamsId}
          isSyncing={isSyncing}
        />
      </div>
    )
  },
)
