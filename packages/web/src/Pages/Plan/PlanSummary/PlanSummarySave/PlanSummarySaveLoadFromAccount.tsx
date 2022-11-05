import { fGet, PlanParams } from '@tpaw/common'
import React, { useState } from 'react'
import { FirebaseUser } from '../../../App/WithFirebaseUser'
import { useSimulation } from '../../../App/WithSimulation'
import { ConfirmAlert } from '../../../Common/Modal/ConfirmAlert'
import { useUser } from '../../../QueryFragments/UserFragment'

export const PlanSummarySaveLoadFromAccount = React.memo(
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
    const { setParams } = useSimulation()
    const [confirm, setConfirm] = useState(false)
    return (
      <>
        <button
          className={`${className} disabled:lighten-2`}
          disabled={!user.plan}
          onClick={() => {
            setConfirm(true)
          }}
        >
          Open Saved Plan
        </button>
        {confirm && (
          <ConfirmAlert
            option1={{
              label: 'Overwrite',
              onClose: () => {
                closeMenu()
                setParams(JSON.parse(fGet(user.plan).params) as PlanParams)
                setConfirm(false)
              },
            }}
            onCancel={() => setConfirm(false)}
          >
            This will overwrite any changes you have made since you last saved
            your plan. Are you sure you want to proceed?
          </ConfirmAlert>
        )}
      </>
    )
  },
)
