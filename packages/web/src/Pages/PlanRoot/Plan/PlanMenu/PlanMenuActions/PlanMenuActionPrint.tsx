import { faPrint } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clsx from 'clsx'
import { useRouter } from 'next/router'
import React, { useMemo } from 'react'
import { useURLUpdater } from '../../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { setPrintOnDoneURL } from '../../PlanPrint/PlanPrint'

export const PlanMenuActionPrint = React.memo(
  ({ className, closeMenu }: { className?: string; closeMenu: () => void }) => {
    const path = useRouter().asPath
    const url = useMemo(() => {
      const result = new URL(path, window.location.origin)
      result.searchParams.set('print', 'true')
      return result
    }, [path])
    const urlUpdater = useURLUpdater()
    const { planParams } = useSimulation()

    // Don't show print until dialogMode is done since print shows results
    // and inputs that are visually not yet available in any other mode.
    const [showInDialogMode, setShowInDialogMode] = React.useState(false)
    return (
      <>
        <Menu.Item
          as="button"
          className={clsx(className, ' context-menu-item')}
          onClick={(e) => {
            if (planParams.dialogPosition !== 'done') {
              // This keeps the menu open (only  on click through, not on keyboard)
              // As of Jun 2023, no solution for keyboard:
              // https://github.com/tailwindlabs/headlessui/discussions/1122
              e.preventDefault()
              setShowInDialogMode(true)
              return
            }
            setPrintOnDoneURL()
            urlUpdater.push(url)
          }}
        >
          <span className="inline-block w-[25px]">
            <FontAwesomeIcon icon={faPrint} />
          </span>{' '}
          Print / Save as PDF
        </Menu.Item>
        <CenteredModal
          show={showInDialogMode}
          onOutsideClickOrEscape={() => setShowInDialogMode(false)}
        >
          <>
            <h2 className=" dialog-heading">Printing Not Yet Available </h2>
            <div className=" dialog-content-div">
              <p className="p-base">
                Printing is only available once you complete the minimum inputs.
              </p>
            </div>
            <div className="dialog-button-div">
              <button
                className=" dialog-button-dark"
                onClick={() => {
                  closeMenu()
                  setShowInDialogMode(false)
                }}
              >
                Close
              </button>
            </div>
          </>
        </CenteredModal>
      </>
    )
  },
)
