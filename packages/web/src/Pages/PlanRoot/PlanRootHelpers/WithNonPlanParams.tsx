import {
  NonPlanParams,
  fGet,
  getDefaultNonPlanParams,
  getZonedTimeFns,
  noCase,
  nonPlanParamsBackwardsCompatibleGuard,
  nonPlanParamsGuard,
  nonPlanParamsMigrate,
} from '@tpaw/common'
import { chain, json, string } from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, {
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { useClientQuery, useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { createContext } from '../../../Utils/CreateContext'
import { AppError } from '../../App/AppError'
import {
  useDefaultErrorHandlerForNetworkCall,
  useSetGlobalError,
} from '../../App/GlobalErrorBoundary'
import { User, useUser } from '../../App/WithUser'
import { WithNonPlanParamsMutation } from './__generated__/WithNonPlanParamsMutation.graphql'

const [Context, useNonPlanParams] = createContext<{
  nonPlanParams: NonPlanParams
  setNonPlanParams: (x: NonPlanParams) => void
}>('NonPlanParams')

export const NonPlanParamsContext = Context

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
  const [nonPlanParams, setNonPlanParamsIn] = useState(
    () =>
      NonPlanParamsLocalStorage.read() ?? getDefaultNonPlanParams(Date.now()),
  )
  const setNonPlanParams = useCallback((x: NonPlanParams) => {
    setNonPlanParamsIn({ ...x, timestamp: Date.now() })
  }, [])
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
    const { setGlobalError } = useSetGlobalError()
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const nonPlanParamsServer = useMemo(
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
    const [nonPlanParams, setNonPlanParamsIn] = useState(nonPlanParamsServer)

    const setNonPlanParams = useCallback((x: NonPlanParams) => {
      setNonPlanParamsIn({ ...x, timestamp: Date.now() })
    }, [])

    const isOutOfSync = useMemo(
      () => !_.isEqual(nonPlanParamsServer, nonPlanParams),
      [nonPlanParams, nonPlanParamsServer],
    )

    const [commit, isRunning] = useMutation<WithNonPlanParamsMutation>(graphql`
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

    const sendToServerEvent = () => {
      commit({
        variables: {
          input: {
            userId: user.id,
            lastUpdatedAt: user.nonPlanParamsLastUpdatedAt,
            nonPlanParams: JSON.stringify(nonPlanParams),
          },
        },
        onCompleted: ({ userSetNonPlanParams }) => {
          switch (userSetNonPlanParams.__typename) {
            case 'UserSuccessResult':
              break
            case 'ConcurrentChangeError':
              setGlobalError(new AppError('concurrentChange'))
              break
            case '%other':
              setGlobalError(new Error())
              break
            default:
              noCase(userSetNonPlanParams)
          }
        },
        onError: (e) => {
          setNonPlanParamsIn(nonPlanParamsServer)
          defaultErrorHandlerForNetworkCall({
            e,
            toast: 'Could not save changes to server.',
          })
        },
      })
    }
    const sendToServerEventRef = useRef(sendToServerEvent)
    sendToServerEventRef.current = sendToServerEvent

    useEffect(() => {
      if (isOutOfSync && !isRunning) {
        sendToServerEventRef.current()
      }
    }, [isOutOfSync, isRunning])

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
