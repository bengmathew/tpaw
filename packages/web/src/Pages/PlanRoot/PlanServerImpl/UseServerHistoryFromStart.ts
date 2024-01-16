import {
  PlanParams,
  PlanParamsHistoryFns,
  assert,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useMemo, useState } from 'react'
import { useRelayEnvironment } from 'react-relay'
import { fetchQuery, graphql } from 'relay-runtime'
import { useAssertConst } from '../../../Utils/UseAssertConst'
import { useUser } from '../../App/WithUser'
import {
  useCurrentPortfolioBalanceGetMonthInfoInWorker,
  useParseAndMigratePlanParamsHistoryInWorker,
} from '../PlanRootHelpers/UsePlanParamsHistoryWorker'
import { PlanParamsHistoryItem } from '../PlanRootHelpers/UseWorkingPlan'
import { useMarketData } from '../PlanRootHelpers/WithMarketData'
import { useIANATimezoneName } from '../PlanRootHelpers/WithNonPlanParams'
import { UseServerHistoryFromStartQuery } from './__generated__/UseServerHistoryFromStartQuery.graphql'

export type ServerHistoryPreBaseInfo = ReturnType<
  typeof useServerHistoryPreBase
>
export const useServerHistoryPreBase = (
  planId: string,
  base: PlanParamsHistoryItem,
) => {
  const { ianaTimezoneName } = useIANATimezoneName()
  const { marketData } = useMarketData()
  const fetchState = useFetchFromServer(planId, base)
  const paramsHistoryFromServer = useParseAndMigratePlanParamsHistoryInWorker(
    fetchState.type === 'fetched' ? fetchState.result : null,
  )
  const [paramsHistoryFromRebase, setParamsHistoryFromRebase] = useState<
    null | { id: string; params: PlanParams }[]
  >(null)

  const planParamsHistory = paramsHistoryFromRebase ?? paramsHistoryFromServer

  const currentPortfolioBalanceByMonthInfo =
    useCurrentPortfolioBalanceGetMonthInfoInWorker(
      planId,
      true,
      planParamsHistory,
    )

  const rebase = (
    currPreBase: { id: string; params: PlanParams }[],
    cutAndBase: { id: string; params: PlanParams }[],
  ) => {
    assert(fGet(_.first(cutAndBase)).id === fGet(_.last(currPreBase)).id)
    const planParamsHistoryUnfiltered = [...currPreBase, ...cutAndBase.slice(1)]
    const { idsToDelete } = PlanParamsHistoryFns.filterForHistoryFromStart({
      ianaTimezoneName,
      planParamsHistory: planParamsHistoryUnfiltered.map((x) => ({
        planParamsChangeId: x.id,
        timestamp: new Date(x.params.timestamp),
      })),
      marketCloses: [
        ...new Set(
          marketData.map((x) => x.dailyStockMarketPerformance.closingTime),
        ).values(),
      ].sort((a, b) => a - b),
    })
    setParamsHistoryFromRebase(
      planParamsHistoryUnfiltered.filter((y) => !idsToDelete.has(y.id)),
    )
  }

  const state = useMemo(() => {
    switch (fetchState.type) {
      case 'fetching':
        return { type: 'fetching' as const }
      case 'fetched':
        return planParamsHistory && currentPortfolioBalanceByMonthInfo
          ? {
              type: 'fetched' as const,
              planParamsHistory,
              currentPortfolioBalanceByMonthInfo,
            }
          : { type: 'fetching' as const }
      case 'failed':
        return { type: 'failed' as const }
      default:
        noCase(fetchState)
    }
  }, [currentPortfolioBalanceByMonthInfo, fetchState, planParamsHistory])
  return { state, rebase }
}

const useFetchFromServer = (planId: string, base: PlanParamsHistoryItem) => {
  const { ianaTimezoneName } = useIANATimezoneName()
  const [startingBase] = useState(base)
  const userId = fGet(useUser()).id
  const [state, setState] = useState<
    | { type: 'fetching'; retryCount: number }
    | { type: 'fetched'; result: readonly { id: string; params: string }[] }
    | { type: 'failed' }
  >({
    type: 'fetching',
    retryCount: 0,
  })
  const relayEnvironment = useRelayEnvironment()
  useEffect(() => {
    if (state.type !== 'fetching') return

    fetchQuery<UseServerHistoryFromStartQuery>(
      relayEnvironment,
      graphql`
        query UseServerHistoryFromStartQuery(
          $userId: ID!
          $planId: String!
          $baseTimestamp: Float!
          $baseId: String!
          $ianaTimezoneName: String!
        ) {
          user(userId: $userId) {
            plan(planId: $planId) {
              id
              planParamsPreBase(
                baseTimestamp: $baseTimestamp
                baseId: $baseId
                ianaTimezoneName: $ianaTimezoneName
              ) {
                id
                params
              }
            }
          }
        }
      `,

      {
        userId,
        planId,
        baseId: startingBase.id,
        baseTimestamp: startingBase.params.timestamp,
        ianaTimezoneName,
      },
    ).subscribe({
      next: (data) =>
        setState({
          type: 'fetched',
          result: fGet(data.user.plan).planParamsPreBase,
        }),
      error: () =>
        setState(
          state.retryCount === 3
            ? { type: 'failed' }
            : { type: 'fetching', retryCount: state.retryCount + 1 },
        ),
    })
  }, [
    relayEnvironment,
    state,
    startingBase.id,
    startingBase.params.timestamp,
    userId,
    planId,
    ianaTimezoneName,
  ])
  useAssertConst([
    startingBase.id,
    startingBase.params.timestamp,
    planId,
    userId,
    relayEnvironment,
  ])

  return state
}
