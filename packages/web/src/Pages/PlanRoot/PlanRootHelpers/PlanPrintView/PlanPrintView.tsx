import { faLeftLong } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { CalendarMonthFns, assert, block, noCase } from '@tpaw/common'
import clsx from 'clsx'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { useEffect, useMemo, useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { appPaths } from '../../../../AppPaths'
import { normalizePlanParams } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { processPlanParams } from '../../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { SimulationResult } from '../../../../UseSimulator/Simulator/Simulator'
import { getSimulatorSingleton } from '../../../../UseSimulator/UseSimulator'
import { asyncEffect } from '../../../../Utils/AsyncEffect'
import { setCSSPageValue } from '../../../../Utils/SetCSSPageValue'
import { useSetGlobalError } from '../../../App/GlobalErrorBoundary'
import { useSystemInfo } from '../../../App/WithSystemInfo'
import { mainPlanColors } from '../../Plan/UsePlanColors'
import { WithPlanResultsChartDataForPDF } from '../../Plan/WithPlanResultsChartData'
import { SimulationResultContext } from '../WithSimulation'
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
import _ from 'lodash'

export type PlanPrintViewProps = {
  fixed: PlanPrintViewArgs['fixed']
  settings: PlanPrintViewArgs['settings']
  simulationResult: SimulationResult | null
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
      useState<SimulationResult | null>(simulationResultIn)

    const planParamsForLink = useMemo(() => {
      const clone = _.cloneDeep(fixed.planParams)
      clone.wealth.portfolioBalance = {
        updatedHere: true,
        amount: fixed.currentPortfolioBalanceAmount,
      }
      return clone
    }, [fixed.currentPortfolioBalanceAmount, fixed.planParams])

    useEffect(() => {
      if (simulationResult) return
      return asyncEffect(async (status) => {
        const nowAsCalendarMonth = CalendarMonthFns.fromTimestamp(
          fixed.timestamp,
          fixed.ianaTimezoneName,
        )
        const planParamsNorm = normalizePlanParams(
          fixed.planParams,
          nowAsCalendarMonth,
        )
        const planParamsProcessed = processPlanParams(
          planParamsNorm,
          fixed.marketData,
        )

        setSimulationResult(
          await getSimulatorSingleton().runSimulations(status, {
            currentPortfolioBalanceAmount: fixed.currentPortfolioBalanceAmount,
            planParamsNorm,
            planParamsProcessed,
            numOfSimulationForMonteCarloSampling:
              fixed.numOfSimulationForMonteCarloSampling,
            randomSeed: fixed.randomSeed,
          }),
        )
      })
    }, [
      fixed.timestamp,
      fixed.ianaTimezoneName,
      fixed.marketData,
      fixed.numOfSimulationForMonteCarloSampling,
      fixed.planParams,
      fixed.randomSeed,
      fixed.currentPortfolioBalanceAmount,
      settings.isServerSidePrint,
      simulationResult,
    ])

    const [shortLink, setShortLink] = useState<URL | null>(null)
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
    const needsShortLink =
      !settings.isServerSidePrint && settings.embeddedLinkType === 'short'
    useEffect(() => {
      if (!needsShortLink || shortLink) return
      const { dispose } = commitGetShortLink({
        variables: {
          input: { params: JSON.stringify(planParamsForLink) },
        },
        onCompleted: ({ createLinkBasedPlan }) => {
          const url = appPaths.link()
          url.searchParams.set('params', createLinkBasedPlan.id)
          setShortLink(url)
        },
        onError: (e) => {
          setGlobalError(e)
        },
      })
      return () => dispose()
    }, [
      commitGetShortLink,
      planParamsForLink,
      needsShortLink,
      setGlobalError,
      shortLink,
    ])

    const linkToEmbed = settings.isServerSidePrint
      ? new URL(settings.linkToEmbed)
      : settings.embeddedLinkType === 'long'
        ? block(() => {
            const url = appPaths.link()
            url.searchParams.set('params', JSON.stringify(planParamsForLink))
            return url
          })
        : shortLink

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
        <SimulationResultContext.Provider value={simulationResult}>
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
                        linkToEmbed={linkToEmbed}
                        settings={settings}
                        planLabel={fixed.planLabel}
                      />
                      <PlanPrintViewInputSection
                        settings={settings}
                        currentPortfolioBalanceAmountInfo={{
                          isEstimate: false,
                          amount: fixed.currentPortfolioBalanceAmount,
                        }}
                      />
                      <PlanPrintViewResultsSection settings={settings} />
                      <PlanPrintViewTasksForThisMonthSection
                        settings={settings}
                      />
                      {simulationResult.args.planParamsNorm.advanced
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
        </SimulationResultContext.Provider>
      </>
    )
  },
)
PlanPrintView.displayName = 'PlanPrintView'
