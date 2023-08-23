import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clsx from 'clsx'
import React, { CSSProperties } from 'react'
import { PlanResultsSidePanelMenuBalanceSheet } from './PlanResutlsSidePanelMenuBalanceSheet'
import { ContextMenu2 } from '../../../../../Common/Modal/ContextMenu2'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../../UsePlanColors'

export const PlanResultsSidePanelMenu = React.memo(
  ({ className, style }: { className?: string; style?: CSSProperties }) => {
    const [showBalanceSheet, setShowBalanceSheet] = React.useState(false)
    const { planParams } = useSimulation()
    const planColors = usePlanColors()

    if (planParams.advanced.strategy !== 'TPAW') return <></>
    return (
      <>
        <ContextMenu2
          align={'right'}
          className={clsx(className, 'w-full')}
          style={{ ...style, backgroundColor: planColors.results.cardBG }}
        >
          <h2 className="font-semibold text-[16px] sm:text-[16px] flex   items-center gap-x-2">
            More
            <FontAwesomeIcon icon={faCaretDown} />
          </h2>
          <Menu.Items className="">
            <Menu.Item
              as="button"
              className="context-menu-item"
              onClick={() => setShowBalanceSheet(true)}
            >
              Balance Sheet
            </Menu.Item>
          </Menu.Items>
        </ContextMenu2>
        <PlanResultsSidePanelMenuBalanceSheet
          show={showBalanceSheet}
          onHide={() => setShowBalanceSheet(false)}
        />
      </>
    )
  },
)
