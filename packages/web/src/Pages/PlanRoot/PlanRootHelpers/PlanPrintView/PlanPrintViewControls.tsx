import {
  faDownload,
  faFilePdf,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { block, fGet, getAppPaths, noCase } from '@tpaw/common'
import clsx from 'clsx'
import React, { useEffect, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { useDefaultErrorHandlerForNetworkCall } from '../../../App/GlobalErrorBoundary'
import { Config } from '../../../Config'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import {
  PlanPrintViewArgs,
  PlanPrintViewArgsServerSide,
  PlanPrintViewSettingsClientSide,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintViewArgs'
import { PlanPrintViewControlsGeneratePDFReportMutation } from './__generated__/PlanPrintViewControlsGeneratePDFReportMutation.graphql'
import { PlanPrintViewSettings } from './PlanPrintViewSettings'

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
        <PlanPrintViewSettings
          className=""
          settings={settings}
          updateSettings={updateSettings}
          disabled={
            state.type === 'generating' ||
            state.type === 'waitingForLinkToEmbed'
          }
        />

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
      </div>
    )
  },
)
