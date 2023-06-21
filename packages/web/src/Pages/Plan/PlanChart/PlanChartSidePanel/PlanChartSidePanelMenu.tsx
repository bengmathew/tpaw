import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clsx from 'clsx'
import React, { CSSProperties } from 'react'
import { useSimulation } from '../../../App/WithSimulation'
import { ContextMenu2 } from '../../../Common/Modal/ContextMenu2'
import { PlanChartSidePanelMenuBalanceSheet } from './PlanChartSidePanelMenuBalanceSheet'

export const PlanChartSidePanelMenu = React.memo(
  ({ className, style }: { className?: string; style?: CSSProperties }) => {
    const [showBalanceSheet, setShowBalanceSheet] = React.useState(false)
    const { params } = useSimulation()
    if (params.plan.advanced.strategy !== 'TPAW') return <></>
    return (
      <>
        <ContextMenu2
          align={'right'}
          className={clsx(className, 'w-full')}
          style={style}
        >
          <h2 className="font-bold text-[16px] sm:text-[20px] flex items-center gap-x-2">
            More
            <FontAwesomeIcon icon={faCaretDown} />
          </h2>
          <Menu.Items className="">
            <Menu.Item
              as="button"
              className="px-4 py-2 text-lg"
              onClick={() => setShowBalanceSheet(true)}
            >
              Balance Sheet
            </Menu.Item>
          </Menu.Items>
        </ContextMenu2>
        <PlanChartSidePanelMenuBalanceSheet
          show={showBalanceSheet}
          onHide={() => setShowBalanceSheet(false)}
        />
      </>
    )
  },
)
