import {
  faDownload,
  faFilePdf,
  faGear,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import { block, fGet, getAppPaths, noCase } from '@tpaw/common'
import clsx from 'clsx'
import React, { ReactNode, useEffect, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { useDefaultErrorHandlerForNetworkCall } from '../../../App/GlobalErrorBoundary'
import { RadioGroupOptionIndicator } from '../../../Common/Inputs/RadioGroupOptionIndicator'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { Config } from '../../../Config'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import {
  PlanPrintViewArgs,
  PlanPrintViewArgsServerSide,
  PlanPrintViewSettingsClientSide,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintViewArgs'
import { PlanPrintViewControlsGeneratePDFReportMutation } from './__generated__/PlanPrintViewControlsGeneratePDFReportMutation.graphql'

export const PlanPrintViewControls = React.memo(
  ({
    linkToEmbed,
    fixedArgs,
    settings,
    updateSettings,
    className,
    style,
  }: {
    linkToEmbed: URL | null
    fixedArgs: PlanPrintViewArgs['fixed']
    settings: PlanPrintViewSettingsClientSide
    updateSettings: (args: PlanPrintViewSettingsControlledClientSide) => void
    className?: string
    style?: React.CSSProperties
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()

    const [showOptions, setShowOptions] = useState(false)

    const [state, setState] = useState<
      | { type: 'idle' }
      | { type: 'waitingForLinkToEmbed' }
      | { type: 'generating' }
      | { type: 'generated'; pdfLink: URL }
    >({ type: 'idle' })

    const [commit] =
      useMutation<PlanPrintViewControlsGeneratePDFReportMutation>(graphql`
        mutation PlanPrintViewControlsGeneratePDFReportMutation(
          $input: GeneratePDFReportInput!
        ) {
          generatePDFReport(input: $input) {
            pdfURL
          }
        }
      `)

    const handleGenerateEvent = (linkToEmbed: URL) => {
      const url = block(() => {
        const params: PlanPrintViewArgsServerSide = {
          fixed: fixedArgs,
          settings: {
            isServerSidePrint: true,
            pageSize: settings.pageSize,
            linkToEmbed: linkToEmbed.toString(),
            alwaysShowAllMonths: settings.alwaysShowAllMonths,
          },
        }
        const appPaths = getAppPaths(new URL(Config.client.urls.deployment))
        const url = new URL(appPaths.serverSidePrint())
        url.searchParams.set('params', JSON.stringify(params))
        return url.toString()
      })
      commit({
        variables: {
          input: {
            url,
            auth: Config.client.debug.authHeader
              ? fGet(Config.client.debug.authHeader.split(' ')[1])
              : null,
            viewportWidth: 1500,
            viewportHeight: 1000,
            devicePixelRatio: 2,
          },
        },
        onCompleted: ({ generatePDFReport: { pdfURL } }) => {
          setState({ type: 'generated', pdfLink: new URL(pdfURL) })
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({
            toast: 'Error generating PDF report.',
            e,
          })
          setState({ type: 'idle' })
        },
      })
      setState({ type: 'generating' })
    }
    const handleGenerateEventRef = React.useRef(handleGenerateEvent)
    handleGenerateEventRef.current = handleGenerateEvent

    useEffect(() => {
      if (state.type === 'waitingForLinkToEmbed' && linkToEmbed)
        handleGenerateEventRef.current(linkToEmbed)
    }, [linkToEmbed, state.type])

    useEffect(() => {
      setState({ type: 'idle' })
    }, [settings])

    const planColors = mainPlanColors

    return (
      <div
        className={clsx(className, 'inline-flex flex-col gap-x-6 ')}
        style={{ ...style }}
      >
        <button
          className=" border-gray-400 text-start  rounded-lg overflow-hidden  text-gray-20  px-4 py-2 "
          disabled={
            state.type === 'waitingForLinkToEmbed' ||
            state.type === 'generating'
          }
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
            <h2 className="">Link Type</h2>
            <h2 className="">:</h2>
            <h2 className="">
              {_embeddedLinkTypeStr(settings.embeddedLinkType)}
            </h2>
          </div>
        </button>

        <div className="flex justify-start mt-2 w-[220px] ">
          <div
            className="w-full rounded-lg"
            style={{
              backgroundColor: planColors.shades.main[12].hex,
              color: planColors.shades.light[3].hex,
            }}
          >
            {block(() => {
              switch (state.type) {
                case 'idle':
                  return (
                    <button
                      className="text-lg flex items-center justify-center gap-x-2 py-3 px-4 w-full "
                      onClick={() => {
                        if (!linkToEmbed) {
                          setState({ type: 'waitingForLinkToEmbed' })
                        } else {
                          handleGenerateEvent(linkToEmbed)
                        }
                      }}
                    >
                      <FontAwesomeIcon icon={faFilePdf} />
                      Generate PDF
                    </button>
                  )
                case 'generating':
                case 'waitingForLinkToEmbed':
                  return (
                    <div className="relative px-4 py-2 w-full">
                      <h2 className="text-lg flex gap-x-2 items-center">
                        <FontAwesomeIcon
                          className="fa-spin"
                          icon={faSpinnerThird}
                        />
                        Generating
                      </h2>
                      <p className="font-font2 lighten text-sm">
                        This may take a few seconds
                      </p>
                    </div>
                  )
                case 'generated':
                  return (
                    <button
                      className="text-lg flex items-center justify-center gap-x-2 py-3 px-4 w-full"
                      onClick={() => {
                        window.open(state.pdfLink.toString(), '_blank')
                      }}
                    >
                      <FontAwesomeIcon icon={faDownload} />
                      Download PDF
                    </button>
                  )
                default:
                  noCase(state)
              }
            })}
          </div>
        </div>

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
      </div>
    )
  },
)

const _pageSizeStr = (pageSize: PlanPrintViewSettingsClientSide['pageSize']) =>
  pageSize === 'A4' ? 'A4' : 'Letter'

const _embeddedLinkTypeStr = (
  x: PlanPrintViewSettingsClientSide['embeddedLinkType'],
) => (x === 'long' ? 'Long' : x === 'short' ? 'Short' : noCase(x))

export const _EditSettings = React.memo(
  ({
    settings,
    updateSettings,
    onDone,
  }: {
    settings: PlanPrintViewSettingsClientSide
    updateSettings: (args: PlanPrintViewSettingsControlledClientSide) => void
    onDone: () => void
  }) => {
    const { pageSize, embeddedLinkType } = settings
    const currUpdatableSettings: PlanPrintViewSettingsControlledClientSide = {
      pageSize: settings.pageSize,
      embeddedLinkType: settings.embeddedLinkType,
    }

    return (
      <div className=" dialog-outer-div">
        <h2 className=" dialog-heading">
          <span className="text-3xl">Settings</span>
        </h2>
        <div className=" dialog-content-div">
          <h2 className="font-bold text-xl mt-10">Paper Size</h2>
          <RadioGroup
            className="max-w-[500px]"
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
          <h2 className="font-bold text-xl mt-10">Link to Plan Type</h2>
          <RadioGroup
            className="max-w-[500px]"
            value={embeddedLinkType}
            onChange={(embeddedLinkType) =>
              updateSettings({ ...currUpdatableSettings, embeddedLinkType })
            }
          >
            <p className="p-base mt-3">
              The PDF report includes a link that recreates this plan. Select
              the type of link to use:
            </p>
            <_RadioOption<PlanPrintViewSettingsClientSide['embeddedLinkType']>
              value={'short'}
              heading={_embeddedLinkTypeStr('short')}
            >
              Shortened link with inputs for the plan stored on the server.
            </_RadioOption>
            <_RadioOption<PlanPrintViewSettingsClientSide['embeddedLinkType']>
              value={'long'}
              heading={_embeddedLinkTypeStr('long')}
            >
              Link containing all the inputs for the plan directly in the link.
            </_RadioOption>
          </RadioGroup>
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
