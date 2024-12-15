import { faLeftLong } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { block, getZonedTimeFns, noCase } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { useEffect, useMemo, useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { appPaths } from '../../../../AppPaths'
import { normalizePlanParams } from '../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { CallRust } from '../../../../Simulator/PlanParamsProcessed/CallRust'
import { processPlanParams } from '../../../../Simulator/PlanParamsProcessed/PlanParamsProcessed'
import { simulateOnServer } from '../../../../Simulator/SimulateOnServer/SimulateOnServer'
import { SimulationResult } from '../../../../Simulator/Simulator/Simulator'
import { setCSSPageValue } from '../../../../Utils/SetCSSPageValue'
import { useSetGlobalError } from '../../../App/GlobalErrorBoundary'
import { useSystemInfo } from '../../../App/WithSystemInfo'
import { getMarketDataForTime } from '../../../Common/GetMarketData'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import { WithPlanResultsChartDataForPDF } from '../../Plan/WithPlanResultsChartData'
import { useMarketData } from '../WithMarketData'
import { SimulationResultInfoContext } from '../WithSimulation'
import { PlanPrintViewAppendixSection } from './PlanPrintViewAppendixSection'
import {
  PlanPrintViewArgs,
  PlanPrintViewSettingsControlledClientSide,
} from './PlanPrintViewArgs'
import { PlanPrintViewBalanceSheetSection } from './PlanPrintViewBalanceSheetSection'
import { PlanPrintViewFrontSection } from './PlanPrintViewFrontSection'
import { PlanPrintViewGenerate } from './PlanPrintViewGenerate'
import { PlanPrintViewInputSection } from './PlanPrintViewInputSection'
import { PlanPrintViewResultsSection } from './PlanPrintViewResultsSection/PlanPrintViewResultsSection'
import { PlanPrintViewSettings } from './PlanPrintViewSettings'
import { PlanPrintViewTasksForThisMonthSection } from './PlanPrintViewTasksForThisMonthSection'
import { PlanPrintViewGetShortLinkMutation } from './__generated__/PlanPrintViewGetShortLinkMutation.graphql'
import { CalendarDayFns } from '../../../../Utils/CalendarDayFns'
import { getPortfolioBalanceEstimationCacheHandlerForDatelessPlan } from '../../../../Simulator/UsePortfolioBalanceEstimationCache'
import { SimulationResult2 } from '../../../../Simulator/UseSimulator'

export type PlanPrintViewProps = {
  fixed: PlanPrintViewArgs['fixed']
  settings: PlanPrintViewArgs['settings']
  simulationResult: SimulationResult2 | null
  updateSettings: (x: PlanPrintViewSettingsControlledClientSide) => void
}

const pixelsToInches = (pixels: number) => pixels / 96
const inchesToPixels = (inches: number) => inches * 96
const pixelsToCM = (pixels: number) => pixelsToInches(pixels) * 2.54
const cmToPixels = (cm: number) => inchesToPixels(cm / 2.54)

export const PlanPrintView = React.memo(
  ({
    fixed,
    settings,
    simulationResult: simulationResultIn,
    updateSettings,
  }: PlanPrintViewProps) => {
    const { pageSize } = settings
    const { setGlobalError } = useSetGlobalError()
    const [simulationResult, setSimulationResult] =
      useState<SimulationResult2 | null>(simulationResultIn)

    const planParamsForLink = useMemo(() => {
      const clone = _.cloneDeep(fixed.planParams)
      clone.wealth.portfolioBalance = clone.datingInfo.isDated
        ? {
            isDatedPlan: true,
            updatedHere: true,
            amount: fixed.currentPortfolioBalanceAmount,
          }
        : {
            isDatedPlan: false,
            amount: fixed.currentPortfolioBalanceAmount,
          }
      return clone
    }, [fixed.currentPortfolioBalanceAmount, fixed.planParams])

    useEffect(() => {
      if (simulationResult) return
      const abortController = new AbortController()
      block(async () => {
        const planParamsNorm = normalizePlanParams(
          fixed.planParams,
          fixed.datingInfo.isDatedPlan
            ? {
                timestamp: fixed.datingInfo.simulationTimestamp,
                calendarDay: CalendarDayFns.fromTimestamp(
                  fixed.datingInfo.simulationTimestamp,
                  fixed.datingInfo.ianaTimezoneName,
                ),
              }
            : {
                // Should not show up in the pdf report.
                timestamp: 0,
                calendarDay: null,
              },
        )

        const simulateOnServerResult = await simulateOnServer(
          abortController.signal,
          fixed.dailyMarketSeriesSrc,
          getPortfolioBalanceEstimationCacheHandlerForDatelessPlan(
            fixed.currentPortfolioBalanceAmount,
          ),
          [
            fixed.percentiles.low,
            fixed.percentiles.mid,
            fixed.percentiles.high,
          ],
          planParamsNorm,
          fixed.numOfSimulationForMonteCarloSampling,
          fixed.randomSeed,
        )
        setSimulationResult({
          ...simulateOnServerResult,
          planParamsNormOfResult: planParamsNorm,
          dailyMarketSeriesSrcOfResult: fixed.dailyMarketSeriesSrc,
          ianaTimezoneNameIfDatedPlanOfResult: fixed.datingInfo.isDatedPlan
            ? fixed.datingInfo.ianaTimezoneName
            : null,
          percentilesOfResult: fixed.percentiles,
          numOfSimulationForMonteCarloSamplingOfResult:
            fixed.numOfSimulationForMonteCarloSampling,
          randomSeedOfResult: fixed.randomSeed,
        })
        // await getSimulatorSingleton().runSimulations(status, {
        //   currentPortfolioBalanceAmount: fixed.currentPortfolioBalanceAmount,
        //   planParamsRust,
        //   marketData: currentMarketData,
        //   planParamsNorm,
        //   planParamsProcessed,
        //   numOfSimulationForMonteCarloSampling:
        //     fixed.numOfSimulationForMonteCarloSampling,
        //   randomSeed: fixed.randomSeed,
        // }),
        // )
      }).catch((e) => {
        if (abortController.signal.aborted) return
        throw e
      })
      return () => abortController.abort()
    }, [
      fixed.datingInfo,
      fixed.dailyMarketSeriesSrc,
      fixed.percentiles,
      fixed.numOfSimulationForMonteCarloSampling,
      fixed.planParams,
      fixed.randomSeed,
      fixed.currentPortfolioBalanceAmount,
      settings.isServerSidePrint,
      simulationResult,
    ])

    const [link, setLink] = useState<string | null>(null)
    const [commitGetShortLink] = useMutation<PlanPrintViewGetShortLinkMutation>(
      graphql`
        mutation PlanPrintViewGetShortLinkMutation(
          $input: CreateLinkBasedPlanInput!
        ) {
          createLinkBasedPlan(input: $input) {
            id
          }
        }
      `,
    )
    const needsLink = !settings.isServerSidePrint && settings.shouldEmbedLink
    useEffect(() => {
      if (!needsLink || link) return
      const { dispose } = commitGetShortLink({
        variables: {
          input: { params: JSON.stringify(planParamsForLink) },
        },
        onCompleted: ({ createLinkBasedPlan }) => {
          const url = appPaths.link()
          url.searchParams.set('params', createLinkBasedPlan.id)
          setLink(url.toString())
        },
        onError: (e) => {
          setGlobalError(e)
        },
      })
      return () => dispose()
    }, [commitGetShortLink, planParamsForLink, needsLink, setGlobalError, link])

    const linkToEmbed = settings.isServerSidePrint
      ? { isProcessing: false, link: settings.linkToEmbed }
      : !settings.shouldEmbedLink
        ? { isProcessing: false, link: null }
        : link
          ? { isProcessing: false, link }
          : { isProcessing: true, link: null }

    const planColors = mainPlanColors

    useEffect(() => {
      setCSSPageValue(`
        size: ${pageSize} portrait; 
        margin-top: 1in;
        margin-bottom: 1in;
        `)
    }, [pageSize])

    const path = useRouter().asPath
    const doneURL = block(() => {
      const url = new URL(path, window.location.origin)
      url.searchParams.delete('pdf-report')
      return url
    })

    const hasSimulationResult = !!simulationResult
    useEffect(() => {
      if (!hasSimulationResult) return
      const w = window as any
      w.__APP_READY_TO_PRINT__ = true
      return () => {
        w.__APP_READY_TO_PRINT__ = false
      }
    }, [hasSimulationResult])

    const { windowSize } = useSystemInfo()
    const simulationIsRunningInfo = useMemo(() => {
      return { isRunning: false as const }
    }, [])

    if (!simulationResult) return <></>

    const pageWidth =
      pageSize === 'A4'
        ? cmToPixels(21)
        : pageSize === 'Letter'
          ? inchesToPixels(8.5)
          : noCase(pageSize)
    // const headerWidth =
    const headerWidth = Math.min(pageWidth, windowSize.width - 10 * 2)

    return (
      <>
        <SimulationResultInfoContext.Provider
          value={{
            simulationResult,
            simulationIsRunningInfo,
          }}
        >
          <WithPlanResultsChartDataForPDF
            planColors={planColors}
            layout="desktop"
            alwaysShowAllMonths={settings.alwaysShowAllMonths}
          >
            <div
              className="print:hidden fixed inset-0 z-0"
              style={{ backgroundColor: planColors.results.bg }}
            />
            {/* relative z-0 makes this a stacking context */}
            <div className="relative z-0">
              {!settings.isServerSidePrint && (
                <div
                  className="print:hidden sticky top-0  z-10 page -mb-16 "
                  style={{
                    backgroundColor: planColors.results.bg,
                  }}
                >
                  <div className="m-auto" style={{ width: `${headerWidth}px` }}>
                    <div className="flex items-center gap-x-4 pt-14 ">
                      <Link
                        className="block btn-dark btn-md "
                        shallow
                        href={doneURL}
                      >
                        <FontAwesomeIcon className="" icon={faLeftLong} /> Done
                      </Link>
                      <h2 className="font-bold text-2xl">PDF Report</h2>
                    </div>
                    <div className="flex justify-between items-end border-b-[2px] border-gray-400 pb-2 mt-2">
                      <h2 className=" mt-5 text-3xl font-bold opacity-40 ">
                        Preview
                      </h2>
                      <PlanPrintViewSettings
                        className=""
                        settings={settings}
                        updateSettings={updateSettings}
                      />
                    </div>
                  </div>
                  <PlanPrintViewGenerate
                    fixedArgs={fixed}
                    linkToEmbed={linkToEmbed}
                    settings={settings}
                  />
                </div>
              )}

              <div className={clsx('px-[10px] print:px-0')}>
                <div
                  className={clsx(
                    'relative z-0',
                    'font-font1 text-black text-[12px]',
                    'w-[calc(100vw-20px)]  overflow-x-scroll print:w-auto print:overflow-x-visible',
                  )}
                >
                  <div
                    className={' m-auto print:m-0'}
                    style={{ width: `${pageWidth}px` }}
                  >
                    <div
                      className={clsx(
                        'flex print:block flex-col items-stretch gap-y-10 ',
                        'pt-20 print:pt-0',
                        'pb-32 print:pb-0',
                        'origin-top',
                      )}
                    >
                      <PlanPrintViewFrontSection
                        linkToEmbed={{
                          needsLink: settings.isServerSidePrint
                            ? !!settings.linkToEmbed
                            : settings.shouldEmbedLink,
                          link: linkToEmbed.link,
                        }}
                        settings={settings}
                        planLabel={fixed.planLabel}
                      />
                      <PlanPrintViewInputSection
                        settings={settings}
                        currentPortfolioBalanceAmount={
                          fixed.currentPortfolioBalanceAmount
                        }
                      />
                      <PlanPrintViewResultsSection settings={settings} />
                      <PlanPrintViewTasksForThisMonthSection
                        settings={settings}
                      />
                      {simulationResult.planParamsNormOfResult.advanced
                        .strategy === 'TPAW' && (
                        <PlanPrintViewBalanceSheetSection settings={settings} />
                      )}
                      <PlanPrintViewAppendixSection settings={settings} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </WithPlanResultsChartDataForPDF>
        </SimulationResultInfoContext.Provider>
      </>
    )
  },
)
PlanPrintView.displayName = 'PlanPrintView'
