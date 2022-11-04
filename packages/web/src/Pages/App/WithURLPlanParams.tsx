import { planParamsGuard } from '@tpaw/common'
import { chain, json } from 'json-guard'
import React, { ReactNode, useMemo } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { useAssertConst } from '../../Utils/UseAssertConst'
import { useURLParam } from '../../Utils/UseURLParam'
import { useURLUpdater } from '../../Utils/UseURLUpdater'
import { ConfirmAlert } from '../Common/Modal/ConfirmAlert'
import { WithURLPlanParamsGetParamsQuery } from './__generated__/WithURLPlanParamsGetParamsQuery.graphql'

export const WithURLPlanParams = React.memo(
  ({ children }: { children?: ReactNode }) => {
    const urlParamsStr = useURLParam('params')

    return urlParamsStr ? (
      <_Handle urlParamsStr={urlParamsStr} />
    ) : (
      <>{children}</>
    )
  },
)

const _Handle = React.memo(({ urlParamsStr }: { urlParamsStr: string }) => {
  const urlParams = useParamsFromStr(urlParamsStr)
  const urlUpdater = useURLUpdater()
  const cleanURL = new URL(window.location.href)
  cleanURL.searchParams.delete('params')

  return urlParams ? (
    <ConfirmAlert
      option1={{
        label: 'Overwrite',
        onClose: () => {
          window.localStorage.setItem('params', JSON.stringify(urlParams))
          urlUpdater.replace(cleanURL)
        },
      }}
      option2={{
        label: 'Ignore Link',
        onOption2: () => urlUpdater.replace(cleanURL),
      }}
      onCancel={null}
    >
      <div className="">
        <p className="">
          You have navigated to this page using a link which contains inputs for
          a plan.
        </p>
        <p className="mt-4">
          Would you like to use the inputs in the link? This will overwrite your
          current inputs.
        </p>
        <p className="mt-4">
          {`If you would like to use the link but not overwrite your current
          inputs, you can open the link in your browser's incognito mode.`}
        </p>
      </div>
    </ConfirmAlert>
  ) : (
    <ConfirmAlert
      option1={{
        label: 'Ignore Inputs in Link',
        onClose: () => urlUpdater.replace(cleanURL),
      }}
      onCancel={null}
    >
      You have navigated to this page using a link which contains inputs to the
      plan, but the link is not valid.
    </ConfirmAlert>
  )
})

const useParamsFromStr = (paramsStr: string) => {
  useAssertConst([paramsStr])
  if (paramsStr.length > 32) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useMemo(() => {
      const guardResult = chain(json, planParamsGuard)(paramsStr)
      return guardResult.error ? null : guardResult.value
    }, [paramsStr])
  } else {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const fullStr = useLazyLoadQuery<WithURLPlanParamsGetParamsQuery>(
      graphql`
        query WithURLPlanParamsGetParamsQuery($linkId: ID!) {
          linkBasedPlan(linkId: $linkId) {
            id
            createdAt
            params
          }
        }
      `,
      { linkId: paramsStr },
    ).linkBasedPlan?.params
    return fullStr? JSON.parse(fullStr) : null
  }
}
