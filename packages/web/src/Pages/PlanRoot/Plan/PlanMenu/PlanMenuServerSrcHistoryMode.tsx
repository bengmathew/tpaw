import {
  faArrowRightFromBracket,
  faCaretDown,
  faCopy,
  faLink,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import React, { useState } from 'react'
import { ContextModal } from '../../../Common/Modal/ContextModal'
import {
  SimulationInfoForHistoryMode,
  SimulationInfoForServerSrc,
  useSimulationInfo,
} from '../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../UsePlanColors'
import { PlanMenuActionModalCopyServer } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyServer'
import { PlanMenuDivider } from './PlanMenuHelpers/PlanMenuDivider'
import { PlanMenuSubMenuRewind } from './PlanMenuSubMenu/PlanMenuSubMenuRewind'
import { PlanMenuActionModalCopyToLink } from './PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyToLink'

export const PlanMenuServerSrcHistoryMode = React.memo(
  ({
    simulationInfoForServerSrc,
    simulationInfoForHistoryMode,
  }: {
    simulationInfoForServerSrc: SimulationInfoForServerSrc
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
  }) => {
    const planColors = usePlanColors()
    const { planParamsId } = useSimulationInfo()
    const { plan, syncState, setRewindTo } = simulationInfoForServerSrc
    const [showCopy, setShowCopy] = useState(false)
    const [showCopyToLink, setShowCopyToLink] = useState(false)
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
                  className={clix('px-3 py-1.5 rounded-lg  m-0.5')}
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
              <Menu.Items className="flex flex-col context-menu-outer-div w-[275px]">
                <Menu.Item>
                  <button
                    className="context-menu-item "
                    onClick={() => setShowCopy(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faCopy} />
                    </span>{' '}
                    Copy as of This Date
                  </button>
                </Menu.Item>
                <Menu.Item>
                  <button
                    className="context-menu-item "
                    onClick={() => setShowCopyToLink(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faLink} />
                    </span>{' '}
                    Copy to Link
                  </button>
                </Menu.Item>

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
          show={showCopy}
          plan={plan}
          onHide={() => setShowCopy(false)}
          hideOnSuccess={false}
          cutAfterId={planParamsId}
          isSyncing={syncState.type !== 'synced'}
        />

        <PlanMenuActionModalCopyToLink
          show={showCopyToLink}
          onDone={() => setShowCopyToLink(false)}
          suggestDateless={false}
        />
      </div>
    )
  },
)
