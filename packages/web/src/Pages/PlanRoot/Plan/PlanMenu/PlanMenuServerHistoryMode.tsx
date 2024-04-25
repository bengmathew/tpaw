import {
  faArrowRightFromBracket,
  faCaretDown,
  faCopy,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import Link from 'next/link'
import React, { useState } from 'react'
import { ContextModal } from '../../../Common/Modal/ContextModal'
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
    const { planParamsId } = useSimulation()
    const { plan, syncState, setRewindTo } = simulationInfoForServerSrc
    const [showCopyModal, setShowCopyModal] = useState(false)
    const label = plan.isMain ? 'Main Plan' : plan.label ?? 'Untitled'

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
                  className={clix('px-3 py-1.5 rounded-lg')}
                  style={{
                    backgroundColor: planColors.results.bg,
                    color: planColors.results.fg,
                  }}
                >
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
                </Menu.Button>
              )}
              <Menu.Items className="flex flex-col py-2.5 rounded-lg">
                <Menu.Item>
                  <button
                    className="context-menu-item "
                    onClick={() => setShowCopyModal(true)}
                  >
                    <span className="context-menu-icon">
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
                  <button
                    className="context-menu-item "
                    onClick={() => setRewindTo(null)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faArrowRightFromBracket} />
                    </span>{' '}
                    Exit History Mode
                  </button>
                </Menu.Item>
              </Menu.Items>
            </ContextModal>
          )}
        </Menu>
        <PlanMenuSubMenuRewind
          simulationInfoForServerSrc={simulationInfoForServerSrc}
          simulationInfoForHistoryMode={simulationInfoForHistoryMode}
        />

        <PlanMenuActionModalCopyServer
          show={showCopyModal}
          plan={plan}
          onHide={() => setShowCopyModal(false)}
          hideOnSuccess={false}
          cutAfterId={planParamsId}
          isSyncing={syncState.type !== 'synced'}
          planParamsForDatingSwitch={null}
        />
      </div>
    )
  },
)
