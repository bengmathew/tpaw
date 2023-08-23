import {
  API,
  NonPlanParams,
  assertFalse,
  fGet,
  letIn,
  noCase,
  planParamsMigrate,
} from '@tpaw/common'
import { chain, json, number, object, success } from 'json-guard'
import _ from 'lodash'
import React, { ReactNode, useEffect, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { Spinner } from '../../Utils/View/Spinner'
import { CenteredModal } from '../Common/Modal/CenteredModal'
import { NonPlanParamsLocalStorage } from '../PlanRoot/PlanRootHelpers/WithNonPlanParams'
import {
  PlanLocalStorage,
  PlanLocalStorageUnmigratedUnsorted,
} from '../PlanRoot/PlanRootLocalMain/PlanLocalStorage'
import { AppPage } from './AppPage'
import { useFirebaseUser } from './WithFirebaseUser'
import {
  UserMergeFromClientLinkPlanInput,
  WithMergeToServerMutation,
  WithMergeToServerMutation$data,
} from './__generated__/WithMergeToServerMutation.graphql'
import Link from 'next/link'
import { appPaths } from '../../AppPaths'

export const WithMergeToServer = React.memo(
  ({ children }: { children: ReactNode }) => {
    const firebaseUser = useFirebaseUser()
    const [localData, setLocalData] = useState(() => {
      return {
        guestPlan: PlanLocalStorage.readUnMigratedUnsorted(),
        linkPlan: MergeToServerLinkPlan.read(),
        nonPlanParams: NonPlanParamsLocalStorage.read(),
      }
    })

    const [savedGuestPlan, setSavedGuestPlan] = useState<{
      label: string
    } | null>(null)

    const handleMergeCompleted = (
      result: WithMergeToServerMutation$data['userMergeFromClient'],
    ) => {
      PlanLocalStorage.clear()
      MergeToServerLinkPlan.clear()
      NonPlanParamsLocalStorage.clear()
      setLocalData({ guestPlan: null, linkPlan: null, nonPlanParams: null })
      if (result.guestPlan) {
        setSavedGuestPlan({
          // Only the first main plan created on the server will not have a
          // label.
          label: fGet(result.guestPlan.label),
        })
      }
    }

    return firebaseUser &&
      (localData.guestPlan || localData.linkPlan || localData.nonPlanParams) ? (
      <_Merge
        userId={firebaseUser.uid}
        localData={localData}
        onDone={handleMergeCompleted}
      />
    ) : (
      <>
        {children}
        <_Alert data={savedGuestPlan} onDone={() => setSavedGuestPlan(null)} />
      </>
    )
  },
)

const _Merge = React.memo(
  ({
    userId,
    localData: { guestPlan, linkPlan, nonPlanParams },
    onDone,
  }: {
    userId: string
    localData: {
      guestPlan: PlanLocalStorageUnmigratedUnsorted | null
      linkPlan: UserMergeFromClientLinkPlanInput | null
      nonPlanParams: NonPlanParams | null
    }
    onDone: (
      result: WithMergeToServerMutation$data['userMergeFromClient'],
    ) => void
  }) => {
    type _State = { type: 'idle' } | { type: 'commit' }

    const [state, setState] = useState<_State>({ type: 'idle' })
    // To avoid double useEffect in dev due to strict mode.
    useEffect(() => setState({ type: 'commit' }), [])

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
        onCompleted: ({ userMergeFromClient }) => onDone(userMergeFromClient),
        onError: (e) => {
          throw e
        },
      })
    }
    const handleCommitRef = useRef(handleCommit)
    handleCommitRef.current = handleCommit

    useEffect(() => {
      if (state.type !== 'commit') return
      handleCommitRef.current()
    }, [state])

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
          <Link href={appPaths.plans()}>
            <h2 className="underline py-2 mt-4 font-semibold">View Your Plans</h2>
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
