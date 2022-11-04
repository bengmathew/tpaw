import {
  faCheck,
  faClipboard,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { errorToast } from '../../../../Utils/CustomToasts'
import { useSimulation } from '../../../App/WithSimulation'
import { Config } from '../../../Config'
import { PlanSummarySaveShortLinkMutation } from './__generated__/PlanSummarySaveShortLinkMutation.graphql'

export const PlanSummarySaveShortLink = React.memo(
  ({ className = '' }: { className?: string }) => {
    const [copied, setCopied] = useState(false)
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
    return (
      <button
        className={`${className}`}
        onClick={() =>
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
              void navigator.clipboard.writeText(href).then(() => {
                setCopied(true)
                window.setTimeout(() => setCopied(false), 1000)
                return null
              })
            },
            onError: () => errorToast(),
          })
        }
      >
        {copied ? (
          <>
            <FontAwesomeIcon className="" icon={faClipboard} /> Copied to
            clipboard{' '}
            <FontAwesomeIcon className="ml-1 font-bold" icon={faCheck} />
          </>
        ) : isMutationRunning ? (
          <>
            <FontAwesomeIcon className="fa-spin" icon={faSpinnerThird} />{' '}
            Generating...
          </>
        ) : (
          'Shortened Link'
        )}
      </button>
    )
  },
)
