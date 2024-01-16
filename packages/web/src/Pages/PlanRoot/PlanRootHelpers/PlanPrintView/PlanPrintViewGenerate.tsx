import {
  faDownload,
  faFilePdf,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, block, fGet, getAppPaths, noCase } from '@tpaw/common'
import React, { useEffect, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { useDefaultErrorHandlerForNetworkCall } from '../../../App/GlobalErrorBoundary'
import { Config } from '../../../Config'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import {
  PlanPrintViewArgs,
  PlanPrintViewArgsServerSide,
  PlanPrintViewSettingsClientSide,
} from './PlanPrintViewArgs'
import { PlanPrintViewGenerateGeneratePDFReportMutation } from './__generated__/PlanPrintViewGenerateGeneratePDFReportMutation.graphql'

export const PlanPrintViewGenerate = React.memo(
  ({
    linkToEmbed,
    fixedArgs,
    settings,
  }: {
    linkToEmbed: URL | null
    fixedArgs: PlanPrintViewArgs['fixed']
    settings: PlanPrintViewSettingsClientSide
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()

    const [state, setState] = useState<
      | { type: 'idle' }
      | { type: 'waitingForLinkToEmbed' }
      | { type: 'generating' }
      | { type: 'generated'; pdfLink: URL }
    >({ type: 'idle' })

    const [commit] =
      useMutation<PlanPrintViewGenerateGeneratePDFReportMutation>(graphql`
        mutation PlanPrintViewGenerateGeneratePDFReportMutation(
          $input: GeneratePDFReportInput!
        ) {
          generatePDFReport(input: $input) {
            pdfURL
          }
        }
      `)

    const cleanupRef = useRef<null | (() => void)>(null)

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
      const { dispose } = commit({
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
          assert(cleanupRef.current)
          cleanupRef.current = null
          setState({ type: 'generated', pdfLink: new URL(pdfURL) })
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({
            toast: 'Error generating PDF report.',
            e,
          })
          assert(cleanupRef.current)
          cleanupRef.current = null
          setState({ type: 'idle' })
        },
      })
      assert(!cleanupRef.current)
      cleanupRef.current = dispose
      setState({ type: 'generating' })
    }
    const handleGenerateEventRef = React.useRef(handleGenerateEvent)
    handleGenerateEventRef.current = handleGenerateEvent

    useEffect(() => {
      if (state.type === 'waitingForLinkToEmbed' && linkToEmbed)
        handleGenerateEventRef.current(linkToEmbed)
    }, [linkToEmbed, state.type])

    useEffect(() => {
      cleanupRef.current?.()
      cleanupRef.current = null
      setState({ type: 'idle' })
    }, [settings])

    useEffect(() => {
      return () => {
        cleanupRef.current?.()
      }
    }, [])

    const planColors = mainPlanColors

    return (
      <div
        className="fixed bottom-5 right-5 rounded-[10px] custom-shadow-md-dark"
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
                      cleanupRef.current?.()
                      cleanupRef.current = null
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
                  <p className="font-font2 lighten text-xs">
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
    )
  },
)
