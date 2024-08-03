import {
  faCaretDown,
  faFolderOpen,
  faPlus,
  faSave,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { assert, fGet, getFullDatedDefaultPlanParams } from '@tpaw/common'
import clsx from 'clsx'
import { DateTime } from 'luxon'
import { useRef, useState } from 'react'
import { appPaths } from '../../../../../AppPaths'
import { errorToast } from '../../../../../Utils/CustomToasts'
import { useURLUpdater } from '../../../../../Utils/UseURLUpdater'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import {
  PlanFileDataFns,
  PLAN_FILE_EXTENSION,
  PlanFileData,
} from '../../../PlanRootFile/PlanFileData'
import { setPlanRootFileOpenWith } from '../../../PlanRootFile/PlanRootFile'
import {
  useIANATimezoneName,
  useNonPlanParams,
} from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForFileSrc,
  SimulationInfoForPlanMode,
} from '../../../PlanRootHelpers/WithSimulation'

export const offlinePlansLabel = 'Save Plans Offline'
export const offlinePlansOpenFileLabel = 'Open File'
export const usePlanMenuSectionOfflinePlans = ({
  info,
  simulationInfoForPlanMode,
}: {
  info:
    | {
        isCurrentlyFile: true
        simulationInfoForFileSrc: SimulationInfoForFileSrc
      }
    | {
        isCurrentlyFile: false
        label: string | null
      }
  simulationInfoForPlanMode: SimulationInfoForPlanMode
}) => {
  const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
  const { undos, redos } = simulationInfoForPlanMode.planParamsUndoRedoStack
  const { ianaTimezoneName, getZonedTime } = useIANATimezoneName()
  // const [isOpen, setIsOpen] = useState(false)
  const [showOverwriteModal, setShowOverwriteLocalModal] = useState(
    false as false | { action: () => void },
  )
  const fileInputElementRef = useRef<HTMLInputElement>(null)
  const [fileInputElementKey, setFileInputElementKey] = useState(0)
  const urlUpdater = useURLUpdater()

  const handleOpen = (filename: string | null, data: PlanFileData) => {
    if (info.isCurrentlyFile) {
      info.simulationInfoForFileSrc.setSrc(filename, data)
    } else {
      setPlanRootFileOpenWith(filename, data)
      urlUpdater.push(appPaths.file())
    }
  }

  const handleSave = () => {
    const now = Date.now()
    const data: PlanFileData = {
      v: 1,
      convertedToFilePlanAtTimestamp: info.isCurrentlyFile
        ? info.simulationInfoForFileSrc.plan.convertedToFilePlanAtTimestamp
        : now,
      lastSavedTimestamp: now,
      planParamsHistory: [...undos, ...redos].map(
        ({ id, change, params, paramsUnmigrated }) => ({
          id,
          change,
          params: paramsUnmigrated ?? params,
        }),
      ),
      reverseHeadIndex: redos.length,
    }
    PlanFileDataFns.download(
      (info.isCurrentlyFile
        ? info.simulationInfoForFileSrc.plan.filename
        : null) ??
        `TPAW Plan (${getZonedTime(now).toLocaleString(DateTime.DATE_MED)})${PLAN_FILE_EXTENSION}`,
      data,
    )
  }

  const menuItems = (
    <div className="context-menu-section">
      <div className="context-menu-section-heading ">
        <div className="flex gap-x-2">
          <h2
            className={clsx(
              ' block',
              !nonPlanParams.showOfflinePlansMenuSection && '',
            )}
          >
            {offlinePlansLabel}{' '}
          </h2>
          <button
            className="underline"
            onClick={(e) => {
              // This keeps the menu open (only on click though, not on keyboard)
              // As of Jun 2023, no solution for keyboard:
              // https://github.com/tailwindlabs/headlessui/discussions/1122
              e.preventDefault()
              setNonPlanParams({
                ...nonPlanParams,
                showOfflinePlansMenuSection:
                  !nonPlanParams.showOfflinePlansMenuSection,
              })
            }}
          >
            {nonPlanParams.showOfflinePlansMenuSection ? 'hide' : 'show'}
          </button>
        </div>
        {nonPlanParams.showOfflinePlansMenuSection && (
          <div className="text-xs lighten ">
            Optional. Alternative to saving plans in account.
          </div>
        )}
      </div>
      {nonPlanParams.showOfflinePlansMenuSection && (
        <div className="">
          <Menu.Item>
            <button className={'context-menu-item'} onClick={handleSave}>
              <span className="context-menu-icon">
                <FontAwesomeIcon className="" icon={faSave} />
              </span>{' '}
              Save to File
            </button>
          </Menu.Item>
          <Menu.Item>
            <button
              className={'context-menu-item'}
              onClick={() => {
                const action = () => fGet(fileInputElementRef.current).click()
                if (
                  info.isCurrentlyFile &&
                  info.simulationInfoForFileSrc.isModified
                ) {
                  setShowOverwriteLocalModal({ action })
                } else {
                  action()
                }
              }}
            >
              <span className="context-menu-icon">
                <FontAwesomeIcon className="" icon={faFolderOpen} />
              </span>{' '}
              {offlinePlansOpenFileLabel}
            </button>
          </Menu.Item>
          <Menu.Item>
            <button
              className={'context-menu-item '}
              onClick={() => {
                const action = () =>
                  handleOpen(
                    null,
                    PlanFileDataFns.getNew(
                      getFullDatedDefaultPlanParams(
                        Date.now(),
                        ianaTimezoneName,
                      ),
                    ),
                  )
                if (
                  info.isCurrentlyFile &&
                  info.simulationInfoForFileSrc.isModified
                ) {
                  setShowOverwriteLocalModal({ action })
                } else {
                  action()
                }
              }}
            >
              <span className="context-menu-icon ">
                <FontAwesomeIcon className="" icon={faPlus} />
              </span>{' '}
              New File
            </button>
          </Menu.Item>
        </div>
      )}
    </div>
  )

  const modals = (
    <>
      <input
        key={fileInputElementKey}
        ref={fileInputElementRef}
        type="file"
        // Note, using .tpaw.txt did not work.
        accept={'.txt'}
        className="hidden"
        multiple={false}
        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        onChange={async (e) => {
          setFileInputElementKey((k) => k + 1)
          const file = fGet(e.target.files)[0]
          const data = await PlanFileDataFns.open(file)
          if (!data) {
            errorToast('Not a valid TPAW file.')
            return
          }
          handleOpen(file.name, data)
        }}
      />
      <CenteredModal
        className=" dialog-outer-div"
        show={!!showOverwriteModal}
        onOutsideClickOrEscape={null}
      >
        <h2 className=" dialog-heading"></h2>
        <div className=" dialog-content-div">
          <p className="p-base">
            You have made changes to this plan since you opened it. If you have
            not saved a copy, you will lose these changes if you continue.
          </p>
        </div>
        <div className=" dialog-button-div">
          <button
            className=" dialog-button-cancel"
            onClick={() => setShowOverwriteLocalModal(false)}
          >
            Cancel
          </button>
          <button
            className=" dialog-button-warning"
            onClick={() => {
              assert(showOverwriteModal)
              setShowOverwriteLocalModal(false)
              showOverwriteModal.action()
            }}
          >
            Continue
          </button>
        </div>
      </CenteredModal>
    </>
  )
  return { menuItems, modals }
}
