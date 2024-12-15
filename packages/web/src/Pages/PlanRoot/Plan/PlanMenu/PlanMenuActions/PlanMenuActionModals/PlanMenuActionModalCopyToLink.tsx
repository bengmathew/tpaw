import { faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, block, fGet, noCase } from '@tpaw/common'
import React, { useEffect, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../../../../../AppPaths'
import { normalizePlanParamsInverse } from '../../../../../../Simulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { Spinner } from '../../../../../../Utils/View/Spinner'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../../App/GlobalErrorBoundary'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useSimulationResultInfo } from '../../../../PlanRootHelpers/WithSimulation'
import { convertToDatelessLabel } from '../../PlanMenuSection/PlanMenuSectionEditLocal'
import { PlanMenuActionModalCopyToLinkMutation } from './__generated__/PlanMenuActionModalCopyToLinkMutation.graphql'

export const PlanMenuActionModalCopyToLink = React.memo(
  ({
    show,
    onDone,
    suggestDateless,
  }: {
    show: boolean
    onDone: () => void
    suggestDateless: 'auto' | false
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body onDone={onDone} suggestDateless={suggestDateless} />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(
  ({
    onDone,
    suggestDateless,
  }: {
    onDone: () => void
    suggestDateless: 'auto' | false
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const [state, setState] = useState<
      | { type: 'idle' }
      | { type: 'gettingLink' }
      | { type: 'gotLink'; link: string }
      | { type: 'copied'; link: string }
    >({ type: 'idle' })

    const onDoneRef = useRef(onDone)
    onDoneRef.current = onDone
    useEffect(() => {
      if (state.type === 'copied') setTimeout(onDoneRef.current, 500)
    }, [state])

    const { simulationResult } = useSimulationResultInfo()
    const { datingInfo } = simulationResult.planParamsNormOfResult

    const [commitGetLink] = useMutation<PlanMenuActionModalCopyToLinkMutation>(
      graphql`
        mutation PlanMenuActionModalCopyToLinkMutation(
          $input: CreateLinkBasedPlanInput!
        ) {
          createLinkBasedPlan(input: $input) {
            id
          }
        }
      `,
    )

    const handleGetLink = () => {
      const params = block(() => {
        const clone = normalizePlanParamsInverse(
          simulationResult.planParamsNormOfResult,
        )
        if (
          clone.wealth.portfolioBalance.isDatedPlan &&
          !clone.wealth.portfolioBalance.updatedHere
        ) {
          assert(datingInfo.isDated)

          clone.timestamp = datingInfo.nowAsTimestamp
          clone.wealth.portfolioBalance = {
            isDatedPlan: true,
            updatedHere: true,
            amount:
              simulationResult.portfolioBalanceEstimationByDated.currentBalance,
          }
        }
        return clone
      })

      commitGetLink({
        variables: { input: { params: JSON.stringify(params) } },
        onCompleted: ({ createLinkBasedPlan }) => {
          const url = appPaths.link()
          url.searchParams.set('params', createLinkBasedPlan.id)
          setState({ type: 'gotLink', link: url.toString() })
        },
        onError: (e) => {
          defaultErrorHandlerForNetworkCall({
            e,
            toast: 'Something went wrong.',
          })
          setState({ type: 'idle' })
        },
      })
      setState({ type: 'gettingLink' })
    }

    return (
      <>
        <h2 className=" dialog-heading">Copy to Link</h2>
        <div className=" dialog-content-div">
          <p className="p-base">
            This creates a copy of the plan and a link to view the copied plan.
            The copy does not contain any plan history. The link will always
            load the plan with these parameter values. Any changes made after loading
            the link will not be reflected back in the link.
          </p>

          {suggestDateless === 'auto' && datingInfo.isDated && (
            <div className="mt-4">
              <p className="p-base">
                <span className="rounded-lg bg-orange-300 px-2 mr-1">
                  {' '}
                  Note
                </span>
                If you are creating this link to share as an example, consider
                converting it to a dateless plan first. Unlike dated plans,
                dateless plans are not tied to the current date and do not
                change over time. You can convert any plan to a dateless plan by
                selecting {`"${convertToDatelessLabel}"`} from the plan menu.
              </p>
            </div>
          )}
          {(state.type === 'gotLink' || state.type === 'copied') && (
            <div
              className=" mt-4 items-center inline-grid gap-x-2"
              style={{ grid: 'auto/auto 1fr' }}
            >
              <h2 className="">Link:</h2>
              <a
                href={state.link}
                target="_blank"
                rel="noreferrer"
                className="underline text-sm lighten-2 whitespace-nowrap text-ellipsis overflow-hidden"
              >
                {state.link}
              </a>
            </div>
          )}
        </div>

        <div className=" dialog-button-div">
          <button className=" dialog-button-cancel" onClick={onDone}>
            Cancel
          </button>
          <button
            className="w-[150px] dialog-button-dark relative"
            onClick={() => {
              switch (state.type) {
                case 'idle':
                  handleGetLink()
                  break
                case 'gettingLink':
                case 'copied':
                  break
                case 'gotLink':
                  void navigator.clipboard.writeText(state.link).then(() => {
                    setState({ type: 'copied', link: state.link })
                    return null
                  })
                  break
                default:
                  noCase(state)
              }
            }}
          >
            {state.type === 'idle' ? (
              <span className="">Create Link</span>
            ) : state.type === 'gettingLink' ? (
              <Spinner />
            ) : state.type === 'gotLink' ? (
              <span className="">Copy Link</span>
            ) : state.type === 'copied' ? (
              <span className="">
                Copied <FontAwesomeIcon className="ml-1" icon={faCheck} />
              </span>
            ) : (
              noCase(state)
            )}
          </button>
        </div>
      </>
    )
  },
)
