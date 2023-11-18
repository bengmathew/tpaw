import {
  NonPlanParams,
  currentNonPlanParamsVersion,
  fGet,
  getDefaultNonPlanParams,
  getZonedTimeFns,
  noCase,
  nonPlanParamsBackwardsCompatibleGuard,
  nonPlanParamsGuard,
  nonPlanParamsMigrate,
} from '@tpaw/common'
import { chain, json, string } from 'json-guard'
import { DateTime } from 'luxon'
import React, {
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { createContext } from '../../../Utils/CreateContext'
import { useAssertConst } from '../../../Utils/UseAssertConst'
import { useDefaultErrorHandlerForNetworkCall } from '../../App/GlobalErrorBoundary'
import { User, useUser } from '../../App/WithUser'
import { WithNonPlanParamsMutation } from './__generated__/WithNonPlanParamsMutation.graphql'

const [Context, useNonPlanParams] = createContext<{
  nonPlanParams: NonPlanParams
  setNonPlanParams: (x: NonPlanParams) => void
}>('NonPlanParams')

export { useNonPlanParams }

export const WithNonPlanParams = ({ children }: { children: ReactNode }) => {
  const user = useUser()
  return user ? (
    <_LoggedIn user={user}>{children}</_LoggedIn>
  ) : (
    <_NotLoggedIn>{children}</_NotLoggedIn>
  )
}

const _NotLoggedIn = React.memo(({ children }: { children: ReactNode }) => {
  const [nonPlanParams, setNonPlanParams] = useState(
    () => NonPlanParamsLocalStorage.read() ?? getDefaultNonPlanParams(),
  )
  useEffect(
    () => NonPlanParamsLocalStorage.write(nonPlanParams),
    [nonPlanParams],
  )
  return (
    <Context.Provider value={{ nonPlanParams, setNonPlanParams }}>
      {children}
    </Context.Provider>
  )
})

const _LoggedIn = React.memo(
  ({ children, user }: { children: ReactNode; user: User }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const nonPlanParams = useMemo(
      () =>
        nonPlanParamsMigrate(
          chain(
            string,
            json,
            nonPlanParamsBackwardsCompatibleGuard,
          )(user.nonPlanParams).force(),
        ),
      [user.nonPlanParams],
    )

    const [commit] = useMutation<WithNonPlanParamsMutation>(graphql`
      mutation WithNonPlanParamsMutation($input: UserSetNonPlanParamsInput!) {
        userSetNonPlanParams(input: $input) {
          __typename
          ... on UserSuccessResult {
            user {
              ...WithUser_user
            }
          }
          ... on ConcurrentChangeError {
            _
          }
        }
      }
    `)

    const setNonPlanParams = useCallback(
      (nonPlanParams: NonPlanParams) => {
        nonPlanParamsGuard(nonPlanParams).force()
        const nonPlanParamsStr = JSON.stringify(nonPlanParams)
        commit({
          variables: {
            input: {
              userId: user.id,
              lastUpdatedAt: user.nonPlanParamsLastUpdatedAt,
              nonPlanParams: nonPlanParamsStr,
            },
          },
          optimisticUpdater: (store) => {
            const userRecord = fGet(store.get(user.id))
            userRecord.setValue(nonPlanParamsStr, 'nonPlanParams')
          },
          onError: (e) => {
            defaultErrorHandlerForNetworkCall({
              e,
              toast: 'Could not save changes to server.',
            })
          },
        })
      },
      [
        commit,
        defaultErrorHandlerForNetworkCall,
        user.id,
        user.nonPlanParamsLastUpdatedAt,
      ],
    )
    useAssertConst([defaultErrorHandlerForNetworkCall])

    const ianaTimezoneName = _getIANATimezoneName(nonPlanParams)

    return (
      <Context.Provider
        // Recreate components if the timezone changes.
        key={ianaTimezoneName}
        value={{ nonPlanParams, setNonPlanParams }}
      >
        {children}
      </Context.Provider>
    )
  },
)

export const useIANATimezoneName = () => {
  const { nonPlanParams } = useNonPlanParams()
  const ianaTimezoneName = _getIANATimezoneName(nonPlanParams)
  const getZonedTime = useMemo(
    () => getZonedTimeFns(ianaTimezoneName),
    [ianaTimezoneName],
  )
  return { ianaTimezoneName, getZonedTime }
}

const _getIANATimezoneName = ({ timezone }: NonPlanParams) =>
  timezone.type === 'auto'
    ? fGet(DateTime.now().zoneName)
    : timezone.type === 'manual'
    ? timezone.ianaTimezoneName
    : noCase(timezone)

export namespace NonPlanParamsLocalStorage {
  export const read = () => {
    const nonPlanParamsStr = localStorage.getItem('nonPlanParams')
    if (!nonPlanParamsStr) return null
    return nonPlanParamsMigrate(
      chain(
        string,
        json,
        nonPlanParamsBackwardsCompatibleGuard,
      )(nonPlanParamsStr).force(),
    )
  }

  export const write = (nonPlanParams: NonPlanParams) => {
    nonPlanParamsGuard(nonPlanParams).force()
    localStorage.setItem('nonPlanParams', JSON.stringify(nonPlanParams))
  }

  export const clear = () => {
    localStorage.removeItem('nonPlanParams')
  }
}
