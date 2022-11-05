import { faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { errorToast } from '../../../../Utils/CustomToasts'
import { Spinner } from '../../../../Utils/View/Spinner'
import { FirebaseUser } from '../../../App/WithFirebaseUser'
import { useSimulation } from '../../../App/WithSimulation'
import { ConfirmAlert } from '../../../Common/Modal/ConfirmAlert'
import { useUser } from '../../../QueryFragments/UserFragment'
import { UserFragment_user$data } from '../../../QueryFragments/__generated__/UserFragment_user.graphql'
import { PlanSummarySaveToAccountMutation } from './__generated__/PlanSummarySaveToAccountMutation.graphql'

export const PlanSummarySaveToAccount = React.memo(
  ({
    className = '',
    firebaseUser,
    closeMenu,
  }: {
    className?: string
    firebaseUser: FirebaseUser
    closeMenu: () => void
  }) => {
    const user = useUser(firebaseUser)

    const [confirm, setConfirm] = useState(false)

    return (
      <>
        <button className={`${className}`} onClick={() => setConfirm(true)}>
          Save
        </button>
        {confirm && (
          <_ConfirmSave
            user={user}
            setConfirm={setConfirm}
            closeMenu={closeMenu}
          />
        )}
      </>
    )
  },
)
const _ConfirmSave = React.memo(
  ({
    className = '',
    user,
    setConfirm,
    closeMenu,
  }: {
    className?: string
    user: UserFragment_user$data
    setConfirm: (x: boolean) => void
    closeMenu: () => void
  }) => {
    // User starting value of hasParams to maitain stability until dialog is
    // closed.
    const [hasPlan] = useState(user.plan !== null)
    const [saved, setSaved] = useState(false)
    const { params } = useSimulation()
    const [commitSaveMutation, isSaveRunning] =
      useMutation<PlanSummarySaveToAccountMutation>(graphql`
        mutation PlanSummarySaveToAccountMutation($input: SetUserPlanInput!) {
          setUserPlan(input: $input) {
            ...UserFragment_user
          }
        }
      `)
    const handleSave = (close: (() => void) | null) => {
      commitSaveMutation({
        variables: {
          input: {
            userId: user.id,
            params: JSON.stringify(params),
          },
        },
        onError: () => {
          close?.()
          errorToast()
        },
        onCompleted: () => {
          setSaved(true)
          window.setTimeout(() => {
            closeMenu()
            close?.()
          }, 1000)
        },
      })
    }
    return (
      <ConfirmAlert
        option1={{
          label: (
            <div className="relative">
              <h2 className={`${isSaveRunning || saved ? 'invisible' : ''}`}>
                {hasPlan ? 'Overwrite' : 'Save'}
              </h2>

              {saved && (
                <div
                  className={`${className} absolute left-[50%] top-[50%] -translate-x-1/2 -translate-y-1/2 text-2xl`}
                >
                  <FontAwesomeIcon
                    className="font-bold text-successFG"
                    icon={faCheck}
                  />
                </div>
              )}
              {isSaveRunning && <Spinner />}
            </div>
          ),
          onBeforeClose: (close) => handleSave(close),
          onClose: () => setConfirm(false),
        }}
        onCancel={() => setConfirm(false)}
      >
        {hasPlan
          ? `You already have a plan saved to your account. Would you like to overwrite it?`
          : 'Save the current plan to your account?'}
      </ConfirmAlert>
    )
  },
)
