import { faCaretDown, faLineColumns } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import React, { CSSProperties } from 'react'
import { ContextModal } from '../../../../../Common/Modal/ContextModal'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../../UsePlanColors'
import { PlanResultsSidePanelMenuBalanceSheet } from './PlanResultsSidePanelMenuBalanceSheet'
import { PlanResultsSidePanelMenuPDFReportItem } from './PlanResultsSidePanelMenuPDFReportItem'
import clsx from 'clsx'

export const PlanResultsSidePanelMenu = React.memo(
  ({ className, style }: { className?: string; style?: CSSProperties }) => {
    const [showBalanceSheet, setShowBalanceSheet] = React.useState(false)
    const { simulationResult } = useSimulationResultInfo()
    const { planParamsNormOfResult } = simulationResult
    const planColors = usePlanColors()

    return (
      <>
        <Menu>
          {({ open }) => (
            <ContextModal
              open={open}
              align={'right'}
              onOutsideClickOrEscape={null}
            >
              {({ ref }) => (
                <Menu.Button
                  ref={ref}
                  className={clsx(
                    className,
                    'w-full',
                    'font-semibold text-[16px] sm:text-[16px] flex   items-center gap-x-2',
                  )}
                  style={{
                    ...style,
                    backgroundColor: planColors.results.cardBG,
                    color: planColors.results.fg,
                  }}
                >
                  More
                  <FontAwesomeIcon icon={faCaretDown} />
                </Menu.Button>
              )}
              <Menu.Items
                className="rounded-lg py-2"
                style={{
                  backgroundColor: planColors.results.cardBG,
                  color: planColors.results.fg,
                }}
              >
                {planParamsNormOfResult.advanced.strategy === 'TPAW' && (
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
            </ContextModal>
          )}
        </Menu>
        <PlanResultsSidePanelMenuBalanceSheet
          show={showBalanceSheet}
          onHide={() => setShowBalanceSheet(false)}
        />
      </>
    )
  },
)
