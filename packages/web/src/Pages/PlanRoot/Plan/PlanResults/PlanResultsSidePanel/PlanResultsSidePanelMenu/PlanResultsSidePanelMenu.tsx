import {
  faCaretDown,
  faFilePdf,
  faLineColumns,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { CSSProperties, useMemo } from 'react'
import { ContextMenu2 } from '../../../../../Common/Modal/ContextMenu2'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../../UsePlanColors'
import { PlanResultsSidePanelMenuBalanceSheet } from './PlanResultsSidePanelMenuBalanceSheet'
import { PlanResultsSidePanelMenuPDFReportItem } from './PlanResultsSidePanelMenuPDFReportItem'

export const PlanResultsSidePanelMenu = React.memo(
  ({ className, style }: { className?: string; style?: CSSProperties }) => {
    const [showBalanceSheet, setShowBalanceSheet] = React.useState(false)
    const { planParams } = useSimulation()
    const planColors = usePlanColors()
    
    return (
      <>
        <ContextMenu2
          align={'right'}
          className={clix(className, 'w-full')}
          style={{
            ...style,
            backgroundColor: planColors.results.cardBG,
            color: planColors.results.fg,
          }}
        >
          <h2 className="font-semibold text-[16px] sm:text-[16px] flex   items-center gap-x-2">
            More
            <FontAwesomeIcon icon={faCaretDown} />
          </h2>
          <Menu.Items
            className="rounded-lg py-2"
            style={{
              backgroundColor: planColors.results.cardBG,
              color: planColors.results.fg,
            }}
          >
          {planParams.advanced.strategy === 'TPAW' && (
              <Menu.Item
                as="button"
                className="context-menu-item"
                onClick={() => setShowBalanceSheet(true)}
              >
                <span className="inline-block w-[30px]">
                  <FontAwesomeIcon icon={faLineColumns} />
                </span>
                Balance Sheet
              </Menu.Item>
            )}
            <PlanResultsSidePanelMenuPDFReportItem />
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
