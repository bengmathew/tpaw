import {
  PlanParams,
  assert,
  fGet,
  planParamsBackwardsCompatibleGuard,
  planParamsMigrate,
} from '@tpaw/common'
import { chain, json, string } from 'json-guard'
import React, { useLayoutEffect, useMemo, useState } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { appPaths } from '../../../AppPaths'
import { useURLParam } from '../../../Utils/UseURLParam'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanRootLinkImpl } from './PlanRootLinkImpl'
import { PlanRootLinkQuery } from './__generated__/PlanRootLinkQuery.graphql'

export const PlanRootLink = React.memo(
  ({ pdfReportInfo }: { pdfReportInfo: SimulationParams['pdfReportInfo'] }) => {
    const paramsStr = useURLParam('params')
    assert(paramsStr !== null)
    const urlUpdater = useURLUpdater()

    const parsed = useMemo(
      () =>
        paramsStr.length === 32
          ? ({
              isLongLink: false,
              linkId: paramsStr,
            } as const)
          : ({
              isLongLink: true,
            } as const),
      [paramsStr],
    )
    useLayoutEffect(() => {
      if (!parsed.isLongLink) return
      const searchParams = new URL(window.location.href).searchParams
      const url = appPaths['convert-long-links']()
      searchParams.forEach((value, key) => url.searchParams.set(key, value))
      urlUpdater.replace(url)
    }, [parsed.isLongLink, urlUpdater])

    if (parsed.isLongLink) return <></>
    return <_Body pdfReportInfo={pdfReportInfo} linkId={parsed.linkId} />
  },
)
const _Body = React.memo(
  ({
    pdfReportInfo,
    linkId,
  }: {
    pdfReportInfo: SimulationParams['pdfReportInfo']
    linkId: string
  }) => {
    const userGQLArgs = useUserGQLArgs()

    const data = useLazyLoadQuery<PlanRootLinkQuery>(
      graphql`
        query PlanRootLinkQuery(
          $userId: ID!
          $includeUser: Boolean!
          $linkId: ID!
        ) {
          ...WithUser_query
          linkBasedPlan(linkId: $linkId) {
            params
          }
        }
      `,
      { ...userGQLArgs, linkId },
    )

    const [startingParams] = useState(() => {
      const result = chain(
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
    })

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
            setStartingParamsOverride(planParams)
            setKey((x) => x + 1)
          }}
          pdfReportInfo={pdfReportInfo}
        />
      </WithUser>
    )
  },
)
