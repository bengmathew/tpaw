import _ from 'lodash'
import { useRouter } from 'next/router'
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useLazyLoadQuery } from 'react-relay'
import { graphql } from 'relay-runtime'
import { createContext } from '../../Utils/CreateContext'
import { useAssertConst } from '../../Utils/UseAssertConst'
import { useURLParam } from '../../Utils/UseURLParam'
import { useURLUpdater } from '../../Utils/UseURLUpdater'
import { AppPage } from '../App/AppPage'
import { useUserGQLArgs } from '../App/WithFirebaseUser'
import { useSimulation } from '../App/WithSimulation'
import { useWindowSize } from '../App/WithWindowSize'
import { Config } from '../Config'
import { planChartLabel } from './PlanChart/PlanChartMainCard/PlanChartLabel'
import { usePlanChartType } from './PlanChart/UsePlanChartType'
import { PlanContent } from './PlanGetStaticProps'
import { PlanHelp } from './PlanHelp'
import {
  isPlanInputType,
  paramsInputTypes,
} from './PlanInput/Helpers/PlanInputType'
import { nextPlanSectionDialogPosition } from './PlanInput/Helpers/PlanSectionDialogPosition'
import { planSectionLabel } from './PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from './PlanInput/Helpers/PlanSectionName'
import { PlanInput } from './PlanInput/PlanInput'
import { planSizing } from './PlanSizing/PlanSizing'
import { PlanSummary } from './PlanSummary/PlanSummary'
import { PlanQuery } from './__generated__/PlanQuery.graphql'

const [PlanContentContext, usePlanContent] =
  createContext<PlanContent>('PlanContent')
export { usePlanContent }

const query = graphql`
  query PlanQuery($userId: ID!, $includeUser: Boolean!) {
    ...UserFragment_query
  }
`

import { noCase, PlanParams } from '@tpaw/common'
import { useMarketData } from '../App/WithMarketData'
import { ConfirmAlert } from '../Common/Modal/ConfirmAlert'
import { WithUser } from '../QueryFragments/UserFragment'
import { PlanChart } from './PlanChart/PlanChart'
import { PlanDialogOverlay } from './PlanDialogOverlay'

export const Plan = React.memo((planContent: PlanContent) => {
  const userGQLArgs = useUserGQLArgs()
  const data = useLazyLoadQuery<PlanQuery>(query, { ...userGQLArgs })
  return (
    <WithUser value={data}>
      <_Plan planContent={planContent} />
    </WithUser>
  )
})

const _Plan = React.memo(({ planContent }: { planContent: PlanContent }) => {
  const simulation = useSimulation()
  const { params, setParams } = simulation

  const isSWR = params.strategy === 'SWR'
  const windowSize = useWindowSize()
  const aspectRatio = windowSize.width / windowSize.height
  const layout =
    aspectRatio > 1.2
      ? 'laptop'
      : windowSize.width <= 700
      ? 'mobile'
      : 'desktop'

  const _sizing = useMemo(
    () => planSizing(layout, windowSize, isSWR),
    [layout, windowSize, isSWR],
  )

  const planChartType = usePlanChartType()
  const chartLabel = planChartLabel(params, planChartType, 'full')

  const state = usePlanState()

  const [transition, setTransition] = useState(() => {
    const x = {
      section: state.section,
      dialogMode: _isDialogMode(state.dialogPosition),
    }
    return { prev: x, target: x, duration: 0 }
  })

  useEffect(() => {
    setTransition((t) => {
      const prev = t.target
      const target = {
        section: state.section,
        dialogMode: _isDialogMode(state.dialogPosition),
      }
      return { prev, target, duration: 300 }
    })
  }, [state, setParams])
  useAssertConst([setParams])

  const [chartDiv, setChartDiv] = useState<HTMLElement | null>(null)
  const isIPhone = window.navigator.userAgent.match(/iPhone/i) !== null

  return (
    <PlanContentContext.Provider value={planContent}>
      <AppPage
        // iPhone viewport height is the max viewport height, but the scroll
        // that results does not properly hide the address and nav bar, so it
        // just does not work. Tested on iOS 15.4. So dont use h-screen, max out
        // based on windowSize.
        className={`${isIPhone ? '' : 'h-screen'} 
        h-screen bg-planBG overflow-hidden`}
        style={{ height: isIPhone ? `${windowSize.height}px` : undefined }}
        title={`Plan
          ${
            planChartType === 'spending-total'
              ? ''
              : ` - View:${_.compact([
                  ...chartLabel.label,
                  chartLabel.subLabel,
                ]).join(' - ')}`
          }
          ${
            state.section === 'summary'
              ? ''
              : ` - ${planSectionLabel(state.section)}`
          }
          - TPAW Planner`}
        curr="plan"
      >
        {/* <PlanWelcome sizing={_sizing.welcome} planTransition={transition} /> */}
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
        <PlanChart
          ref={setChartDiv}
          layout={layout}
          sizing={_sizing.chart}
          planTransition={transition}
          section={state.section}
        />
        <PlanSummary
          section={state.section}
          sizing={_sizing.summary}
          planTransition={transition}
        />
        {(!params.warnedAbout14to15Converstion ||
          !params.warnedAbout16to17Converstion) && (
          <ConfirmAlert
            title={'Migration to New Risk Inputs'}
            option1={{
              label: 'Close',
              onClose: () => {
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  clone.warnedAbout14to15Converstion = true
                  clone.warnedAbout16to17Converstion = true
                  return clone
                })
              },
            }}
            onCancel={null}
          >
            <p className="p-base">
              The planner has been updated to use different inputs for risk.
              Your previous risk inputs have been migrated to the new version.
              The mapping is not exact, so please review the inputs in the risk
              section.
            </p>
          </ConfirmAlert>
        )}
        <PlanDialogOverlay chartDiv={chartDiv} />
      </AppPage>
    </PlanContentContext.Provider>
  )
})

function usePlanState() {
  const { params, setParams, paramsExt } = useSimulation()
  const { asYFN, withdrawalStartYear } = paramsExt
  const withdrawalStartYearAYFN = asYFN(withdrawalStartYear)

  const urlSection = useURLSection()

  // It's important to keep section and dialogMode together so they change
  // together. Otherwise there will end up being multiple planTransitions when
  // we intend for there to be only one.
  const [state, setState] = useState({
    section: urlSection,
    dialogPosition: params.dialogPosition,
  })
  useEffect(() => {
    setState((prev) => ({
      section: urlSection,
      dialogPosition:
        prev.dialogPosition !== 'done' &&
        urlSection === 'summary' &&
        prev.section === prev.dialogPosition
          ? nextPlanSectionDialogPosition(
              prev.dialogPosition,
              withdrawalStartYearAYFN,
            )
          : prev.dialogPosition,
    }))
    // ignore withdrawalStartYearAYFN
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlSection])

  useEffect(() => {
    setParams((params) => {
      if (params.dialogPosition === state.dialogPosition) return params
      const clone = _.cloneDeep(params)
      clone.dialogPosition = state.dialogPosition
      return clone
    })
  }, [setParams, state.dialogPosition])

  useEffect(() => {
    setState((prev) =>
      params.dialogPosition === prev.dialogPosition
        ? prev
        : { section: prev.section, dialogPosition: params.dialogPosition },
    )
  }, [params.dialogPosition])

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
  const path = useRouter().asPath
  return useCallback(
    (section: PlanSectionName) => {
      const url = new URL(`${Config.client.urls.app()}${path}`)
      url.pathname = section === 'summary' ? '/plan' : `/plan/${section}`
      return url
    },
    [path],
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
