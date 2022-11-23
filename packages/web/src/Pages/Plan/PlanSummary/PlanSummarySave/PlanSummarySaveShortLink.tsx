import {
  faCheck,
  faClipboard,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assertFalse, noCase } from '@tpaw/common'
import React, { useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { errorToast } from '../../../../Utils/CustomToasts'
import { useSimulation } from '../../../App/WithSimulation'
import { Config } from '../../../Config'
import { PlanSummarySaveShortLinkMutation } from './__generated__/PlanSummarySaveShortLinkMutation.graphql'

type _State =
  | { type: 'idle' }
  | { type: 'showCopied' }
  | { type: 'gotLink'; href: string }
export const PlanSummarySaveShortLink = React.memo(
  ({
    className = '',
    closeMenu,
  }: {
    className?: string
    closeMenu: () => void
  }) => {
    const [state, setState] = useState<_State>({ type: 'idle' })
    const { params } = useSimulation()
    const [commitMutation, isMutationRunning] =
      useMutation<PlanSummarySaveShortLinkMutation>(graphql`
        mutation PlanSummarySaveShortLinkMutation(
          $input: CreateLinkBasedPlanInput!
        ) {
          createLinkBasedPlan(input: $input) {
            id
          }
        }
      `)

    const handleClick = () => {
      switch (state.type) {
        case 'idle':
          commitMutation({
            variables: { input: { params: JSON.stringify(params) } },
            onCompleted: (result) => {
              const href = new URL(
                Config.client.urls.app(
                  `/plan?${new URLSearchParams({
                    params: result.createLinkBasedPlan.id,
                  }).toString()}`,
                ),
              ).toString()
              setState({ type: 'gotLink', href })
            },
            onError: () => errorToast(),
          })
          break
        case 'gotLink':
          void navigator.clipboard.writeText(state.href).then(() => {
            setState({ type: 'showCopied' })
            window.setTimeout(() => closeMenu(), 1000)
            return null
          })
          break
        case 'showCopied':
          assertFalse()
        default:
          noCase(state)
      }
    }
    return (
      <button
        className={`${className}`}
        disabled={isMutationRunning || state.type === 'showCopied'}
        onClick={handleClick}
      >
        {state.type === 'showCopied' ? (
          <>
            <FontAwesomeIcon className="mr-2" icon={faClipboard} />Copied{' '}
            <FontAwesomeIcon className="ml-1 font-bold" icon={faCheck} />
          </>
        ) : state.type === 'gotLink' ? (
          <>
            <FontAwesomeIcon className="mr-2" icon={faClipboard} />Copy to
            Clipboard
          </>
        ) : isMutationRunning ? (
          <>
            <FontAwesomeIcon className="fa-spin mr-2" icon={faSpinnerThird} />
            Generating...
          </>
        ) : state.type === 'idle' ? (
          'Shortened Link'
        ) : (
          noCase(state)
        )}
      </button>
    )
  },
)
