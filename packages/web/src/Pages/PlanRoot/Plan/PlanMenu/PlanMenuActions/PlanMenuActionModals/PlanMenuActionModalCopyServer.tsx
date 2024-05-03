import { block, fGet } from '@tpaw/common'
import Link from 'next/link'
import React, { useEffect, useRef, useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { appPaths } from '../../../../../../AppPaths'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { User, useUser } from '../../../../../App/WithUser'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { setPlansOnDoneURL } from '../../../../../Plans/Plans'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'
import { PlanMenuActionModalCopyServerCopyMutation } from './__generated__/PlanMenuActionModalCopyServerCopyMutation.graphql'

export const PlanMenuActionModalCopyServer = React.memo(
  ({
    plan,
    show,
    onHide,
    hideOnSuccess,
    cutAfterId,
    isSyncing,
  }: {
    plan: User['plans'][0]
    show: boolean
    onHide: () => void
    hideOnSuccess: boolean
    cutAfterId: string | null
    isSyncing: boolean
  }) => {
    return (
      <CenteredModal
        className="w-[450px] dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          plan={plan}
          onHide={onHide}
          hideOnSuccess={hideOnSuccess}
          cutAfterId={cutAfterId}
          isSyncing={isSyncing}
        />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    plan,
    onHide,
    hideOnSuccess,
    cutAfterId,
    isSyncing,
  }: {
    plan: User['plans'][0]
    onHide: () => void
    hideOnSuccess: boolean
    cutAfterId: string | null
    isSyncing: boolean
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const [result, setResult] = React.useState<{ slug: string } | null>(null)
    const [label, setLabel] = useState(null as string | null)
    const user = fGet(useUser())

    const [copyMutation, isCopyRunning] =
      useMutation<PlanMenuActionModalCopyServerCopyMutation>(graphql`
        mutation PlanMenuActionModalCopyServerCopyMutation(
          $input: UserPlanCopyInput!
        ) {
          userPlanCopy(input: $input) {
            plan {
              slug
              ...PlanWithoutParamsFragment
            }
            user {
              ...WithUser_user
            }
          }
        }
      `)

    const handleCopy = (label: string) => {
      copyMutation({
        variables: {
          input: {
            userId: user.id,
            planId: plan.id,
            label,
            cutAfterId,
          },
        },
        onCompleted: ({ userPlanCopy }) => {
          if (hideOnSuccess) {
            onHide()
          } else {
            setResult(userPlanCopy.plan)
          }
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({
            e,
            toast: 'Error copying plan.',
          })
          onHide()
        },
      })
    }
    const handleCopyRef = useRef(handleCopy)
    handleCopyRef.current = handleCopy

    useEffect(() => {
      if (!label || isSyncing) return
      handleCopyRef.current(label)
    }, [isSyncing, label])

    const isRunning = isCopyRunning || isSyncing

    return !result ? (
      <PlanMenuActionModalLabelInput
        title="Copy Plan"
        initialLabel={plan.label ? `Copy of ${plan.label}` : ''}
        buttonLabel="Copy Plan"
        onCancel={onHide}
        onAction={setLabel}
        isRunning={isRunning}
      />
    ) : (
      <>
        <h2 className=" dialog-heading">Successfully Copied Plan</h2>
        <div className=" dialog-content-div">
          <Link
            href={block(() => {
              const url = appPaths['alt-plan']()
              url.searchParams.set('plan', result.slug)
              return url
            })}
            onClick={() => setPlansOnDoneURL()}
          >
            <h2 className="underline py-2 mt-2">Switch to Copied Plan</h2>
          </Link>
          <Link href={appPaths.plans()} onClick={() => setPlansOnDoneURL()}>
            <h2 className="underline py-2">View All Plans</h2>
          </Link>
        </div>
        <div className=" dialog-button-div">
          <button className="btn-md btn-dark" onClick={onHide}>
            Close
          </button>
        </div>
      </>
    )
  },
)
