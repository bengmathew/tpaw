import {
  PlanParams,
  assert,
  fGet,
  planParamsBackwardsCompatibleGuard,
  planParamsMigrate,
} from '@tpaw/common'
import { chain, json, string } from 'json-guard'
import React, { useEffect, useMemo, useState } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { useURLParam } from '../../../Utils/UseURLParam'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanRootLinkImpl } from './PlanRootLinkImpl'
import { PlanRootLinkQuery } from './__generated__/PlanRootLinkQuery.graphql'
import { set } from 'lodash'

export const PlanRootLink = React.memo(
  ({ pdfReportInfo }: { pdfReportInfo: SimulationParams['pdfReportInfo'] }) => {
    const paramsStr = useURLParam('params')
    assert(paramsStr !== null)
    const [startingState] = useState(() =>
      paramsStr.length === 32
        ? ({
            shortOrLongLink: 'shortLink',
            queryArgs: { linkId: paramsStr, includeLink: true },
          } as const)
        : ({
            shortOrLongLink: 'longLink',
            params: chain(
              string,
              json,
              planParamsBackwardsCompatibleGuard,
            )(paramsStr).force(),
            queryArgs: { linkId: '', includeLink: false },
          } as const),
    )

    const userGQLArgs = useUserGQLArgs()

    const [fetchKey, setFetchKey] = useState(0)
    const data = useLazyLoadQuery<PlanRootLinkQuery>(
      graphql`
        query PlanRootLinkQuery(
          $userId: ID!
          $includeUser: Boolean!
          $linkId: ID!
          $includeLink: Boolean!
        ) {
          ...WithUser_query

          linkBasedPlan(linkId: $linkId) @include(if: $includeLink) {
            params
          }
        }
      `,
      { ...userGQLArgs, ...startingState.queryArgs },
      { fetchKey },
    )
    useEffect(() => {
      const interval = window.setInterval(
        () => setFetchKey((x) => x + 1),
        1000 * 60,
      )
      return () => window.clearInterval(interval)
    }, [])

    const startingParams = useMemo(() => {
      const result =
        startingState.shortOrLongLink === 'longLink'
          ? startingState.params
          : chain(
              string,
              json,
              planParamsBackwardsCompatibleGuard,
            )(fGet(data.linkBasedPlan).params).force()
      const migrated = planParamsMigrate(result)
      assert(
        !migrated.wealth.portfolioBalance.isDatedPlan ||
          migrated.wealth.portfolioBalance.updatedHere,
      )
      return result
    }, [startingState, data])

    const [key, setKey] = useState(0)
    const [startingParamsOverride, setStartingParamsOverride] = useState(
      null as PlanParams | null,
    )

    return (
      <WithUser userFragmentOnQueryKey={userGQLArgs.includeUser ? data : null}>
        <PlanRootLinkImpl
          key={key}
          startingParams={startingParams}
          startingParamsOverride={startingParamsOverride}
          reset={(planParams: PlanParams | null) => {
            if (planParams) setStartingParamsOverride(planParams)
            setKey((x) => x + 1)
          }}
          pdfReportInfo={pdfReportInfo}
        />
      </WithUser>
    )
  },
)
