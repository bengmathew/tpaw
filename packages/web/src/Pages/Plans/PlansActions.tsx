import {
  faArrowDownToLine,
  faArrowUp,
  faCaretDown,
  faCopy,
  faEraser,
  faTag,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import React from 'react'
import { User } from '../App/WithUser'
import { ContextModal } from '../Common/Modal/ContextModal'
import { PlanMenuActionModalCopyServer } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCopyServer'
import { PlanMenuActionModalDelete } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalDelete'
import { PlanMenuActionModalEditLabel } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalEditLabel'
import { PlanMenuActionModalResetServer } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalResetServer'
import { PlanMenuActionModalSetAsMain } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalSetAsMain'
import clsx from 'clsx'

export const PlansActions = React.memo(
  ({
    className = '',
    plan,
  }: {
    className?: string
    plan: User['plans'][0]
  }) => {
    const [showEditLabelModal, setShowEditLabelModal] = React.useState(false)
    const [showSetAsMainModal, setShowSetAsMainModal] = React.useState(false)
    const [showCopyModal, setShowCopyModal] = React.useState(false)
    const [showDelete, setShowDelete] = React.useState(false)
    const [showReset, setShowReset] = React.useState(false)
    return (
      <>
        <Menu>
          {({ open }) => (
            <ContextModal
              open={open}
              align="right"
              onOutsideClickOrEscape={null}
            >
              {({ ref }) => (
                <Menu.Button ref={ref} className={clsx(className)}>
                  Actions <FontAwesomeIcon icon={faCaretDown} />
                </Menu.Button>
              )}
              <Menu.Items className="py-2.5 min-w-[250px]">
                <Menu.Item
                  as="button"
                  className="context-menu-item"
                  onClick={() => setShowEditLabelModal(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faTag} />
                  </span>{' '}
                  Edit Label
                </Menu.Item>
                {!plan.isMain && (
                  <Menu.Item
                    as="button"
                    className="context-menu-item"
                    onClick={() => setShowSetAsMainModal(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faArrowUp} />
                    </span>{' '}
                    Set as Main
                  </Menu.Item>
                )}
                <Menu.Item
                  as="button"
                  className="context-menu-item"
                  onClick={() => setShowCopyModal(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faCopy} />
                  </span>{' '}
                  Copy 
                </Menu.Item>
                <Menu.Item
                  as="button"
                  className="context-menu-item text-errorFG "
                  onClick={() => setShowReset(true)}
                >
                  <span className="context-menu-icon">
                    <FontAwesomeIcon icon={faEraser} />
                  </span>{' '}
                  Reset
                </Menu.Item>
                {!plan.isMain && (
                  <Menu.Item
                    as="button"
                    className="context-menu-item text-errorFG"
                    onClick={() => setShowDelete(true)}
                  >
                    <span className="context-menu-icon">
                      <FontAwesomeIcon icon={faTrash} />
                    </span>{' '}
                    Delete
                  </Menu.Item>
                )}
              </Menu.Items>
            </ContextModal>
          )}
        </Menu>
        <PlanMenuActionModalEditLabel
          show={showEditLabelModal}
          plan={plan}
          onHide={() => setShowEditLabelModal(false)}
        />
        <PlanMenuActionModalSetAsMain
          show={showSetAsMainModal}
          plan={plan}
          onHide={() => setShowSetAsMainModal(false)}
          switchToMainPlanOnSuccess={false}
          isSyncing={false}
        />
        <PlanMenuActionModalCopyServer
          show={showCopyModal}
          plan={plan}
          onHide={() => setShowCopyModal(false)}
          hideOnSuccess
          cutAfterId={null}
          isSyncing={false}
        />
        <PlanMenuActionModalDelete
          show={showDelete}
          plan={plan}
          onHide={() => setShowDelete(false)}
          toPlansOnSuccess={false}
          isSyncing={false}
        />
        <PlanMenuActionModalResetServer
          show={showReset}
          plan={plan}
          onHide={() => setShowReset(false)}
          isSyncing={false}
          reloadOnSuccess={null}
        />
      </>
    )
  },
)
