import { faGear } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup, Switch } from '@headlessui/react'
import { noCase } from '@tpaw/common'
import clsx from 'clsx'
import React, { ReactNode, useState } from 'react'
import { RadioGroupOptionIndicator } from '../../../Common/Inputs/RadioGroupOptionIndicator'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import {
  PlanPrintViewSettingsClientSide,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintViewArgs'
import { SwitchAsToggle } from '../../../Common/Inputs/SwitchAsToggle'
import { SwitchAsCheckBox } from '../../../Common/Inputs/SwitchAsCheckBox'

export const PlanPrintViewSettings = React.memo(
  ({
    settings,
    updateSettings,
    className,
  }: {
    settings: PlanPrintViewSettingsClientSide
    updateSettings: (args: PlanPrintViewSettingsControlledClientSide) => void
    className?: string
  }) => {
    const [showOptions, setShowOptions] = useState(false)
    const planColors = mainPlanColors

    return (
      <>
        <button
          className={clsx(
            className,
            ' border-gray-400 text-start  rounded-lg overflow-hidden  text-gray-20  px-4 py-2 ',
          )}
          onClick={() => setShowOptions(true)}
          style={{
            backgroundColor: planColors.shades.light[2].hex,
          }}
        >
          <h2 className="text-base font-bold flex items-center justify-between ">
            <div className="">
              <FontAwesomeIcon className="mr-2" icon={faGear} /> Settings
            </div>
          </h2>
          <div
            className="text-sm mt-1 inline-grid gap-x-3 gap-y-1"
            style={{ grid: 'auto/auto auto 1fr' }}
          >
            <h2 className="">Paper Size</h2>
            <h2 className="">:</h2>
            <h2 className="">{_pageSizeStr(settings.pageSize)}</h2>
            <h2 className="">Add Link</h2>
            <h2 className="">:</h2>
            <h2 className="">{settings.shouldEmbedLink ? 'Yes' : 'No'}</h2>
          </div>
        </button>

        <CenteredModal
          show={showOptions}
          onOutsideClickOrEscape={() => setShowOptions(false)}
        >
          <_EditSettings
            settings={settings}
            updateSettings={updateSettings}
            onDone={() => setShowOptions(false)}
          />
        </CenteredModal>
      </>
    )
  },
)

const _pageSizeStr = (pageSize: PlanPrintViewSettingsClientSide['pageSize']) =>
  pageSize === 'A4' ? 'A4' : 'Letter'

const _EditSettings = React.memo(
  ({
    settings,
    updateSettings,
    onDone,
  }: {
    settings: PlanPrintViewSettingsClientSide
    updateSettings: (args: PlanPrintViewSettingsControlledClientSide) => void
    onDone: () => void
  }) => {
    const { pageSize, shouldEmbedLink } = settings
    const currUpdatableSettings: PlanPrintViewSettingsControlledClientSide = {
      pageSize: settings.pageSize,
      shouldEmbedLink: settings.shouldEmbedLink,
    }

    return (
      <div className=" dialog-outer-div">
        <h2 className=" dialog-heading">
          <span className="text-3xl">Settings</span>
        </h2>
        <div className=" dialog-content-div max-w-[500px]">
          <h2 className="font-bold text-xl mt-10">Paper Size</h2>
          <RadioGroup
            className=""
            value={pageSize}
            onChange={(pageSize) =>
              updateSettings({ ...currUpdatableSettings, pageSize })
            }
          >
            <_RadioOption<PlanPrintViewSettingsClientSide['pageSize']>
              value={'Letter'}
              heading={_pageSizeStr('Letter')}
            >
              8.5 in x 11 in. Commonly used the United States, Canada, and a few
              other countries.
            </_RadioOption>
            <_RadioOption<PlanPrintViewSettingsClientSide['pageSize']>
              value={'A4'}
              heading={_pageSizeStr('A4')}
            >
              21 cm x 29.7 cm. Commonly used in most countries outside of the
              United States and Canada.
            </_RadioOption>
          </RadioGroup>
          <h2 className="font-bold text-xl mt-10">Add Link</h2>
          <Switch.Group>
            <div className="flex  gap-x-2 mt-4">
              <SwitchAsCheckBox
                className="mr-1 shrink-0 mt-1"
                checked={shouldEmbedLink}
                setChecked={(shouldEmbedLink) =>
                  updateSettings({ ...currUpdatableSettings, shouldEmbedLink })
                }
              />
              <Switch.Label className="p-base cursor-pointer">
                Add a link. This creates a copy of the plan and adds a link in
                the pdf to view the copied plan.
              </Switch.Label>
            </div>
          </Switch.Group>

          <div className=" dialog-button-div">
            <button className=" dialog-button-dark" onClick={onDone}>
              Done
            </button>
          </div>
        </div>
      </div>
    )
  },
)

const _RadioOption = <T,>({
  className,
  value,
  heading,
  children,
}: {
  className?: string
  value: T
  heading: ReactNode
  children: ReactNode
}) => {
  return (
    <RadioGroup.Option<React.ElementType, T>
      value={value}
      className={clsx(
        className,
        'flex items-center gap-x-2 text-start cursor-pointer w-full mt-3',
      )}
    >
      {({ checked }) => (
        <div
          className={
            clsx()
            // ' border rounded-lg  p-2 mt-2',
            // checked && 'bg-gray-100',
          }
        >
          <h2 className="font-semibold flex items-center gap-x-2">
            <RadioGroupOptionIndicator
              className="text-gray-700"
              size="base"
              selected={checked}
            />
            {heading}
          </h2>
          <p className="p-base pl-6">{children}</p>
        </div>
      )}
    </RadioGroup.Option>
  )
}
