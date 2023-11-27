import {
  API,
  NonPlanParams,
  fGet,
  letIn,
  planParamsMigrate,
} from '@tpaw/common'
import { User } from 'firebase/auth'
import { chain, json, number, object, success } from 'json-guard'
import _ from 'lodash'
import Link from 'next/link'
import React, { ReactNode, useEffect, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../AppPaths'
import { Spinner } from '../../Utils/View/Spinner'
import { CenteredModal } from '../Common/Modal/CenteredModal'
import { NonPlanParamsLocalStorage } from '../PlanRoot/PlanRootHelpers/WithNonPlanParams'
import {
  PlanLocalStorage,
  PlanLocalStorageUnmigratedUnsorted,
} from '../PlanRoot/PlanRootLocalMain/PlanLocalStorage'
import { AppPage } from './AppPage'
import { useSetGlobalError } from './GlobalErrorBoundary'
import { useFirebaseUser } from './WithFirebaseUser'
import {
  UserMergeFromClientLinkPlanInput,
  WithMergeToServerMutation,
  WithMergeToServerMutation$data,
} from './__generated__/WithMergeToServerMutation.graphql'

const _getLocalData = () => ({
  guestPlan: PlanLocalStorage.readUnMigratedUnsorted(),
  linkPlan: MergeToServerLinkPlan.read(),
  nonPlanParams: NonPlanParamsLocalStorage.read(),
})

const _clearLocalData = () => {
  PlanLocalStorage.clear()
  MergeToServerLinkPlan.clear()
  NonPlanParamsLocalStorage.clear()
}

const _getNeedsToMerge = (localData: {
  guestPlan: PlanLocalStorageUnmigratedUnsorted | null
  linkPlan: UserMergeFromClientLinkPlanInput | null
  nonPlanParams: NonPlanParams | null
}) => !!(localData.guestPlan || localData.linkPlan || localData.nonPlanParams)

export const WithMergeToServer = React.memo(
  ({ children }: { children: ReactNode }) => {
    const firebaseUser = useFirebaseUser()
    return firebaseUser ? (
      <_WithMergeToServer firebaseUser={firebaseUser}>
        {children}
      </_WithMergeToServer>
    ) : (
      <>{children}</>
    )
  },
)

// A specific component if the user is logged in makes it easy to manage the state
// of merging/non merging in the presense or absence of the user.
const _WithMergeToServer = React.memo(
  ({ children, firebaseUser }: { children: ReactNode; firebaseUser: User }) => {
    const [state, setState] = useState<
      | { isMerging: true }
      | { isMerging: false; savedGuestPlan: { label: string } | null }
    >(() =>
      _getNeedsToMerge(_getLocalData())
        ? { isMerging: true }
        : { isMerging: false, savedGuestPlan: null },
    )

    const handleMergeCompleted = (
      result: WithMergeToServerMutation$data['userMergeFromClient'],
    ) => {
      setState({
        isMerging: false,
        savedGuestPlan: result.guestPlan
          ? {
              // Only the first main plan created on the server will not have a
              // label.
              label: fGet(result.guestPlan.label),
            }
          : null,
      })
    }

    return state.isMerging ? (
      <_Merge userId={firebaseUser.uid} onDone={handleMergeCompleted} />
    ) : (
      <>
        {children}
        <_Alert
          data={state.savedGuestPlan}
          onDone={() => setState({ isMerging: false, savedGuestPlan: null })}
        />
      </>
    )
  },
)

const _Merge = React.memo(
  ({
    userId,
    onDone,
  }: {
    userId: string
    onDone: (
      result: WithMergeToServerMutation$data['userMergeFromClient'],
    ) => void
  }) => {
    const { setGlobalError } = useSetGlobalError()

    // To avoid double useEffect in dev due to strict mode.
    const [isInitialized, setIsInitialized] = useState(false)

    const [commit] = useMutation<WithMergeToServerMutation>(graphql`
      mutation WithMergeToServerMutation($input: UserMergeFromClientInput!) {
        userMergeFromClient(input: $input) {
          guestPlan {
            label
            ...PlanWithoutParamsFragment
          }
          linkPlan {
            label
            ...PlanWithoutParamsFragment
          }
        }
      }
    `)

    const handleCommit = () => {
      const localData = _getLocalData()
      if (!_getNeedsToMerge(localData)) {
        // This can happen if another tab got the lock first and merged.
        onDone({ guestPlan: null, linkPlan: null })
        return
      }
      const { guestPlan, linkPlan, nonPlanParams } = localData
      commit({
        variables: {
          input: {
            userId,
            guestPlan: guestPlan
              ? {
                  planParamsHistory: letIn(
                    new Map(
                      guestPlan.planParamsPostBaseUnmigratedUnsorted.map(
                        (x) => [x.id, x],
                      ),
                    ),
                    (map) =>
                      _.sortBy(
                        guestPlan.planParamsPostBaseUnmigratedUnsorted,
                        (x) => planParamsMigrate(x.params).timestamp,
                      )
                        .map((x) => fGet(map.get(x.id)))
                        .map((x) => ({
                          id: x.id,
                          params: JSON.stringify(x.params),
                          change: JSON.stringify(x.change),
                        })),
                  ),
                  reverseHeadIndex: guestPlan.reverseHeadIndex,
                }
              : null,
            linkPlan,
            nonPlanParams: nonPlanParams ? JSON.stringify(nonPlanParams) : null,
          },
        },
        onCompleted: ({ userMergeFromClient }) => {
          _clearLocalData()
          onDone(userMergeFromClient)
        },
        onError: (e) => {
          // Cannot really proceed meaningfull after this. User should reload.
          // So don't use defaultErrorHandlerForNetworkCall. Set to crashed page.
          setGlobalError(e)
        },
      })
    }
    const handleCommitRef = useRef(handleCommit)
    handleCommitRef.current = handleCommit

    useEffect(() => {
      if (!isInitialized) {
        setIsInitialized(true)
      } else {
        const subState = {
          abortController: null as AbortController | null,
          resolve: null as (() => void) | null,
        }
        subState.abortController = new AbortController()
        // Locking is necessary to prevent two tabs from merging at the same.
        // This can happen when a new user signs in through email, which opens
        // a new tab, resulting in two tabs being open and needing to merge.
        void window.navigator.locks.request(
          'mergeToServer',
          { mode: 'exclusive', signal: subState.abortController.signal },
          () => {
            subState.abortController = null
            handleCommitRef.current()
            return new Promise<void>((resolve) => {
              subState.resolve = resolve
            })
          },
        )
        return () => {
          // This means component has unmounted, because isInitialized once
          // true, never goes back to false.
          if (subState.abortController) {
            subState.abortController.abort('Component unmounted.')
          }
          if (subState.resolve) {
            subState.resolve()
          }
        }
      }
    }, [isInitialized])

    return (
      <AppPage
        className="page h-screen flex flex-col justify-center items-center"
        title={''}
      >
        <Spinner size="text-4xl" />
      </AppPage>
    )
  },
)

const _Alert = React.memo(
  ({
    data,
    onDone,
  }: {
    data: { label: string } | null
    onDone: () => void
  }) => {
    const [lastNonNullData, setLastNonNullData] = useState(data)
    useEffect(() => {
      if (data !== null) setLastNonNullData(data)
    }, [data])

    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={data !== null}
        onOutsideClickOrEscape={null}
        onLeave={onDone}
      >
        <h2 className=" dialog-heading">Guest Plan Moved To Account</h2>
        <div className=" dialog-content-div">
          <p className="p-base mt-2">
            You had a guest plan that was saved on the browser. This has been
            moved to your account and given the label{' '}
            <span className="font-bold">{`"${
              lastNonNullData?.label ?? ''
            }."`}</span>
          </p>
          <Link href={appPaths.plans()} onClick={() => onDone()}>
            <h2 className="underline py-2 mt-4 font-semibold">
              View Your Plans
            </h2>
          </Link>
        </div>
        <div className=" dialog-button-div">
          <button className=" dialog-button-dark" onClick={() => onDone()}>
            Close
          </button>
        </div>
      </CenteredModal>
    )
  },
)

export namespace MergeToServerLinkPlan {
  const key = 'WithMergeToServer_LinkPlanInput'
  export const write = (input: UserMergeFromClientLinkPlanInput) => {
    window.localStorage.setItem(
      key,
      JSON.stringify({ timestamp: Date.now(), input }),
    )
  }

  export const read = (): UserMergeFromClientLinkPlanInput | null => {
    const str = window.localStorage.getItem(key)
    if (!str) return null
    const shapeCheck = chain(
      json,
      object({
        timestamp: number,
        input: (x) => success<unknown>(x),
      }),
    )(str)

    const fail = () => {
      clear()
      return null
    }

    if (
      shapeCheck.error ||
      shapeCheck.value.timestamp < Date.now() - 1000 * 60 * 20
    )
      return fail()
    const typeCheck = API.UserMergeFromClient.parts.linkPlan(
      shapeCheck.value.input,
    ).error
    return typeCheck
      ? fail()
      : (shapeCheck.value.input as UserMergeFromClientLinkPlanInput)
  }
  export const clear = () => window.localStorage.removeItem(key)
}
