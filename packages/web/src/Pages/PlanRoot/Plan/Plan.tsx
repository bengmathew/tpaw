import { PlanParams, block, noCase } from '@tpaw/common'
import clsx from 'clsx'
import getIsMobile from 'is-mobile'
import _ from 'lodash'
import { useRouter } from 'next/router'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useURLParam } from '../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { AppPage } from '../../App/AppPage'
import {
  WithScrollbarWidth,
  useScrollbarWidth,
} from '../../App/WithScrollbarWidth'
import { useWindowSize } from '../../App/WithWindowSize'
import { ChartPointerPortal } from '../../Common/Chart/ChartComponent/ChartPointerPortal'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'
import { PlanChartPointer } from './PlanChartPointer/PlanChartPointer'
import { PlanContact } from './PlanContact/PlanContact'
import { PlanDialogOverlay } from './PlanDialogOverlay'
import { PlanHelp } from './PlanHelp'
import {
  isPlanInputType,
  paramsInputTypes,
} from './PlanInput/Helpers/PlanInputType'
import { nextPlanSectionDialogPosition } from './PlanInput/Helpers/PlanSectionDialogPosition'
import { planSectionLabel } from './PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from './PlanInput/Helpers/PlanSectionName'
import { PlanInput } from './PlanInput/PlanInput'
import { PlanMenu } from './PlanMenu/PlanMenu'
import { PlanMigrationWarnings } from './PlanMigrationWarnings'
import { PlanPrint } from './PlanPrint/PlanPrint'
import { PlanResults } from './PlanResults/PlanResults'
import { planResultsChartLabel } from './PlanResults/PlanResultsChartCard/PlanResultsChartLabel'
import { usePlanResultsChartType } from './PlanResults/UsePlanResultsChartType'
import { planSizing } from './PlanSizing/PlanSizing'
import { PlanSummary } from './PlanSummary/PlanSummary'
import { usePlanColors } from './UsePlanColors'
import { WithPlanResultsChartData } from './WithPlanResultsChartData'

export const Plan = React.memo(() => {
  return (
    <WithScrollbarWidth>
      <_Plan />
    </WithScrollbarWidth>
  )
})
Plan.displayName = 'Plan'

const _Plan = React.memo(() => {
  const { planParams, simulationInfoByMode, simulationInfoBySrc } =
    useSimulation()

  const isSWR = planParams.advanced.strategy === 'SWR'
  const isTallMenu =
    simulationInfoBySrc.src === 'localMain' ||
    simulationInfoByMode.mode === 'history'
  const { windowSize, windowWidthName } = useWindowSize()
  const scrollbarWidth = useScrollbarWidth()
  const aspectRatio = windowSize.width / windowSize.height
  const layout =
    windowWidthName === 'xs'
      ? 'mobile'
      : aspectRatio > 1.2
      ? 'laptop'
      : 'desktop'

  const _sizing = useMemo(
    () => planSizing(layout, windowSize, scrollbarWidth, isSWR, isTallMenu),
    [layout, windowSize, scrollbarWidth, isSWR, isTallMenu],
  )

  const planChartType = usePlanResultsChartType()
  const chartLabel = planResultsChartLabel(planParams, planChartType)

  const state = usePlanState()

  const [transition, setTransition] = useState(() => {
    const x = {
      section: state.section,
      dialogMode: _isDialogMode(state.dialogPosition),
      chartHover: false,
    }
    return { prev: x, target: x, duration: 0 }
  })
  const [chartPointerPortal] = useState(() => new ChartPointerPortal())
  const [chartHover, setChartHover] = useState(false)

  useEffect(() => {
    setTransition((t) => {
      const prev = t.target
      const target = {
        section: state.section,
        dialogMode: _isDialogMode(state.dialogPosition),
        chartHover: false,
      }
      return { prev, target, duration: 300 }
    })
  }, [state])

  const [chartDiv, setChartDiv] = useState<HTMLElement | null>(null)
  const isIPhone = window.navigator.userAgent.match(/iPhone/i) !== null
  const planColors = usePlanColors()
  const showPrint = useURLParam('print') === 'true'
  const planLabel = block(() => {
    switch (simulationInfoBySrc.src) {
      case 'link':
        return 'From Link'
      case 'localMain':
        return null
      case 'server':
        return simulationInfoBySrc.plan.isMain
          ? null
          : simulationInfoBySrc.plan.label
      default:
        noCase(simulationInfoBySrc)
    }
  })
  return (
    <WithPlanResultsChartData
      planSizing={_sizing}
      planTransitionState={transition.target}
    >
      {showPrint && <PlanPrint planLabel={planLabel ?? null} />}
      <AppPage
        // iPhone viewport height is the max viewport height, but the scroll
        // that results does not properly hide the address and nav bar, so it
        // just does not work. Tested on iOS 15.4. So dont use h-screen for
        // iPhone, max out based on windowSize.
        // className={`${i  sIPhone ? '' : 'h-screen'} bg-planBG overflow-hidden`}
        className={clsx(
          showPrint && 'hidden',
          planColors.pageBG,
          'overflow-hidden',
          !isIPhone && 'h-screen',
          // state.section === 'print' && 'hidden',
        )}
        style={{ height: isIPhone ? `${windowSize.height}px` : undefined }}
        title={_.compact([
          `Plan ${planLabel ? `(${planLabel}) ` : ''}`,
          ...(showPrint
            ? ['Print']
            : [
                planChartType === 'spending-total'
                  ? undefined
                  : `View: ${_.compact([
                      ...chartLabel.label.full,
                      chartLabel.subLabel,
                    ]).join(' > ')}`,
                state.section === 'summary'
                  ? undefined
                  : planSectionLabel(state.section),
              ]),
          'TPAW Planner',
        ]).join(' - ')}
      >
        {({ setDarkHeader }) => (
          <>
            {paramsInputTypes.map((type) => (
              <PlanInput
                key={type}
                layout={layout}
                sizing={_sizing.input}
                planTransition={transition}
                planInputType={type}
              />
            ))}
            <PlanHelp sizing={_sizing.help} planTransition={transition} />

            <PlanSummary
              section={state.section}
              sizing={_sizing.summary}
              planTransition={transition}
            />
            <PlanMenu sizing={_sizing.menu} planTransition={transition} />
            <PlanContact sizing={_sizing.contact} planTransition={transition} />
            <PlanMigrationWarnings />
            <PlanDialogOverlay chartDiv={chartDiv} />
            <div
              className="absolute z-40 inset-0 pointer-events-none bg-black/60"
              style={{
                transitionProperty: 'opacity',
                transitionDuration: '300ms',
                opacity: `${getIsMobile() && chartHover ? 1 : 0}`,
              }}
            />
            {/* Place this before PlanResults, so sizing changes propagate to it first. */}
            <PlanChartPointer
              sizing={_sizing.pointer}
              planTransition={transition}
              chartPointerPortal={chartPointerPortal}
            />
            <PlanResults
              ref={setChartDiv}
              layout={layout}
              sizing={_sizing.chart}
              planTransition={transition}
              section={state.section}
              chartPointerPortal={chartPointerPortal}
              onChartHover={(hover) => {
                if (getIsMobile()) setDarkHeader(hover)
                setChartHover(hover)
              }}
              chartHover={chartHover}
            />
          </>
        )}
      </AppPage>
    </WithPlanResultsChartData>
  )
})
_Plan.displayName = '_Plan'

function usePlanState() {
  const { planParams, updatePlanParams, planParamsExt, simulationInfoByMode } =
    useSimulation()
  const { isFutureSavingsAllowed } = planParamsExt

  const urlSection = useURLSection()

  // It's important to keep section and dialogMode together so they change
  // together. Otherwise there will end up being multiple planTransitions when
  // we intend for there to be only one. (Do batched updates in React 18 obviate
  // this?)
  const [state, setState] = useState({
    section: urlSection,
    dialogPosition: planParams.dialogPosition,
  })
  const handleURLSectionChange = (section: typeof urlSection) =>
    setState({
      section,
      dialogPosition:
        state.dialogPosition !== 'done' &&
        urlSection === 'summary' &&
        state.section === state.dialogPosition
          ? nextPlanSectionDialogPosition(
              state.dialogPosition,
              isFutureSavingsAllowed,
            )
          : state.dialogPosition,
    })
  const handleURLSectionChangeRef = useRef(handleURLSectionChange)
  handleURLSectionChangeRef.current = handleURLSectionChange
  useEffect(() => handleURLSectionChangeRef.current(urlSection), [urlSection])

  const handleDialogPosition = (
    dialogPosition: PlanParams['dialogPosition'],
  ) => {
    if (simulationInfoByMode.mode === 'history') return
    if (planParams.dialogPosition === dialogPosition) return
    updatePlanParams('setDialogPosition', dialogPosition)
  }
  const handleDialogPositionRef = useRef(handleDialogPosition)
  handleDialogPositionRef.current = handleDialogPosition
  useEffect(
    () => handleDialogPositionRef.current(state.dialogPosition),
    [state.dialogPosition],
  )

  useEffect(() => {
    setState((prev) =>
      planParams.dialogPosition === prev.dialogPosition
        ? prev
        : { section: prev.section, dialogPosition: planParams.dialogPosition },
    )
  }, [planParams.dialogPosition])

  return state
}

function useURLSection() {
  const urlUpdater = useURLUpdater()
  const getSectionURL = useGetSectionURL()

  const urlSectionStr = useURLParam('section') ?? ''
  const urlSection: PlanSectionName =
    isPlanInputType(urlSectionStr) || urlSectionStr === 'help'
      ? urlSectionStr
      : 'summary'

  // Keep the URL up to date with urlSection. This is
  // needed because the empty section in the url can be interpreted as
  // 'welcome', which should reflect in the URL.
  if (urlSectionStr.length > 0 && urlSectionStr !== urlSection)
    urlUpdater.replace(getSectionURL(urlSection))

  return urlSection
}

export const useGetSectionURL = () => {
  const { planPaths } = useSimulation()
  const path = useRouter().asPath
  return useCallback(
    (section: PlanSectionName) => {
      const result = section === 'summary' ? planPaths() : planPaths[section]()
      new URL(path, window.location.origin).searchParams.forEach((value, key) =>
        result.searchParams.set(key, value),
      )
      return result
    },
    [planPaths, path],
  )
}

const _isDialogMode = (dialogPosition: PlanParams['dialogPosition']) => {
  switch (dialogPosition) {
    case 'age':
    case 'current-portfolio-balance':
    case 'future-savings':
    case 'income-during-retirement':
      return true
    case 'show-results':
    case 'show-all-inputs':
    case 'done':
      return false
    default:
      noCase(dialogPosition)
  }
}
