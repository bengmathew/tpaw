import React, { useEffect, useState } from 'react'
import { useLazyLoadQuery } from 'react-relay'
import { graphql } from 'relay-runtime'
import * as uuid from 'uuid'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanFileData } from './PlanFileData'
import { PlanRootFileImpl } from './PlanRootFileImpl'
import { PlanRootFileOpen } from './PlanRootFileOpen'
import { PlanRootFileQuery } from './__generated__/PlanRootFileQuery.graphql'

let _planRootFileOpenWith: {
  filename: string | null
  originalData: PlanFileData
} | null = null
export const setPlanRootFileOpenWith = (
  filename: string | null,
  data: PlanFileData,
) => {
  _planRootFileOpenWith = { filename, originalData: data }
}

export const PlanRootFile = React.memo(
  ({ pdfReportInfo }: { pdfReportInfo: SimulationParams['pdfReportInfo'] }) => {
    const userGQLArgs = useUserGQLArgs()
    const data = useLazyLoadQuery<PlanRootFileQuery>(
      graphql`
        query PlanRootFileQuery($userId: ID!, $includeUser: Boolean!) {
          ...WithUser_query
        }
      `,
      { ...userGQLArgs },
    )

    const [src, setSrc] = useState<{
      key: number
      filename: string | null
      originalData: PlanFileData
      // When we reset, we need original data as well for isModified check,
      // So we keep both around. Client should use resetData ?? originalData.
      resetData: PlanFileData | null
    } | null>(
      _planRootFileOpenWith
        ? { key: 0, ..._planRootFileOpenWith, resetData: null }
        : null,
    )

    useEffect(() => {
      _planRootFileOpenWith = null
    }, [])

    return (
      <WithUser userFragmentOnQueryKey={userGQLArgs.includeUser ? data : null}>
        {src ? (
          <PlanRootFileImpl
            key={src.key}
            pdfReportInfo={pdfReportInfo}
            src={src}
            setSrc={(filename, data) =>
              setSrc({
                key: src.key + 1,
                originalData: data,
                resetData: null,
                filename,
              })
            }
            reset={(planParams) =>
              setSrc({
                key: src.key + 1,
                filename: src.filename,
                originalData: src.originalData,
                resetData: planParams
                  ? {
                      ...src.originalData,
                      planParamsHistory: [
                        {
                          id: uuid.v4(),
                          change: { type: 'start', value: null },
                          params: planParams,
                        },
                      ],
                    }
                  : null,
              })
            }
          />
        ) : (
          <PlanRootFileOpen
            onDone={(filename, data) =>
              setSrc({ key: 0, originalData: data, resetData: null, filename })
            }
          />
        )}
      </WithUser>
    )
  },
)
