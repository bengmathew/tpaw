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
import { useSimulation, WithSimulation } from '../App/WithSimulation'
import { useWindowSize } from '../App/WithWindowSize'
import { Config } from '../Config'
import { PlanChart } from './PlanChart/PlanChart'
import { planChartLabel } from './PlanChart/PlanChartMainCard/PlanChartLabel'
import { usePlanChartType } from './PlanChart/UsePlanChartType'
import { PlanContent } from './PlanGetStaticProps'
import {
  isPlanInputType,
  paramsInputTypes
} from './PlanInput/Helpers/PlanInputType'
import { planSectionLabel } from './PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from './PlanInput/Helpers/PlanSectionName'
import { PlanInput } from './PlanInput/PlanInput'
import { PlanResults } from './PlanResults'
import { planSizing } from './PlanSizing/PlanSizing'
import { PlanSummary } from './PlanSummary/PlanSummary'
import { PlanWelcome } from './PlanWelcome'
import { PlanQuery } from './__generated__/PlanQuery.graphql'

const [PlanContentContext, usePlanContent] =
  createContext<PlanContent>('PlanContent')
export { usePlanContent }

const query = graphql`
  query PlanQuery($userId: ID!, $includeUser: Boolean!) {
    ...UserFragment_query
  }
`

import { WithChartData } from '../App/WithChartData'
import { WithUser } from '../QueryFragments/UserFragment'

export const Plan = React.memo((planContent: PlanContent) => {
  const userGQLArgs = useUserGQLArgs()
  const data = useLazyLoadQuery<PlanQuery>(query, { ...userGQLArgs })
  return (
    <WithUser value={data}>
      <WithSimulation>
        <WithChartData>
          <_Plan planContent={planContent} />
        </WithChartData>
      </WithSimulation>
    </WithUser>
  )
})

const _Plan = React.memo(({ planContent }: { planContent: PlanContent }) => {
  const simulation = useSimulation()
  const { params, setParams } = simulation
  const windowSize = useWindowSize()
  const aspectRatio = windowSize.width / windowSize.height
  const layout =
    aspectRatio > 1.2
      ? 'laptop'
      : windowSize.width <= 700
      ? 'mobile'
      : 'desktop'

  const _sizing = useMemo(
    () => planSizing(layout, windowSize),
    [layout, windowSize],
  )

  const planChartType = usePlanChartType()
  const chartLabel = planChartLabel(params, planChartType, 'full')

  const state = usePlanState()

  const [transition, setTransition] = useState(() => ({
    prev: state,
    target: state,
    duration: 0,
  }))

  useEffect(() => {
    setTransition((t) => {
      const prev = t.target
      const target = state
      const duration =
        target.dialogMode && !prev.dialogMode
          ? 10 // reset
          : target.dialogMode || prev.dialogMode
          ? 1000
          : 300
      return { prev, target: state, duration }
    })
  }, [state, setParams])
  useAssertConst([setParams])

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
              : state.section === 'results' && state.dialogMode
              ? ' - Preliminary Results'
              : ` - ${planSectionLabel(state.section, params.strategy)}`
          }
          - TPAW Planner`}
        curr="plan"
      >
        <PlanWelcome sizing={_sizing.welcome} planTransition={transition} />
        {paramsInputTypes.map((type) => (
          <PlanInput
            key={type}
            layout={layout}
            sizing={_sizing.input}
            planTransition={transition}
            planInputType={type}
          />
        ))}
        <PlanResults sizing={_sizing.results} planTransition={transition} />
        <PlanChart
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
      </AppPage>
    </PlanContentContext.Provider>
  )
})

function usePlanState() {
  const { params, setParams } = useSimulation()

  const urlSection = useURLSection()

  // It's important to keep section and dialogMode together so they change
  // together. Otherwise there will end up being multiple planTransitions when
  // we intend for there to be only one.
  const [state, setState] = useState({
    section: urlSection,
    dialogMode: params.dialogMode && _isDialogInputStage(urlSection),
  })
  useEffect(() => {
    setState((prev) => ({
      section: urlSection,
      dialogMode:
        urlSection === 'welcome' ||
        (prev.dialogMode && _isDialogInputStage(urlSection)),
    }))
  }, [urlSection])

  useEffect(() => {
    setParams((params) => {
      if (params.dialogMode === state.dialogMode) return params
      const clone = _.cloneDeep(params)
      clone.dialogMode = state.dialogMode
      return clone
    })
  }, [setParams, state.dialogMode])

  return state
}

function useURLSection() {
  const { params } = useSimulation()
  const urlUpdater = useURLUpdater()
  const getSectionURL = useGetSectionURL()

  const urlSectionStr = useURLParam('section') ?? ''
  const urlSection: PlanSectionName =
    isPlanInputType(urlSectionStr) || urlSectionStr === 'results'
      ? urlSectionStr
      : params.dialogMode
      ? 'welcome'
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
    (section:  PlanSectionName) => {
      const url = new URL(`${Config.client.urls.app()}${path}`)
      url.pathname =
        section === 'summary' || section === 'welcome'
          ? '/plan'
          : `/plan/${section}`
      return url
    },
    [path],
  )
}

const _isDialogInputStage = (section: PlanSectionName | 'done') =>
  [
    'welcome',
    'age',
    'current-portfolio-balance',
    'future-savings',
    'income-during-retirement',
    'results',
  ].includes(section)
