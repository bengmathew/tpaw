import {
  faCheck,
  faCopy,
  faLink,
  faSpinnerThird,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { captureException } from '@sentry/nextjs'
import { fGet, noCase } from '@tpaw/common'
import clsx from 'clsx'
import cloneJSON from 'fast-json-clone'
import React, { ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../../../../AppPaths'
import { errorToast } from '../../../../../Utils/CustomToasts'
import { CurrentPortfolioBalance } from '../../../PlanRootHelpers/CurrentPortfolioBalance'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionCopyToLinkShortLinkMutation } from './__generated__/PlanMenuActionCopyToLinkShortLinkMutation.graphql'
import { AppError } from '../../../../App/AppError'
import { useDefaultErrorHandlerForNetworkCall } from '../../../../App/GlobalErrorBoundary'

export const PlanMenuActionCopyToLink = React.memo(
  ({ className, closeMenu }: { className?: string; closeMenu: () => void }) => {
    const [state, setState] = useState<{ type: 'idle' } | { type: 'pickType' }>(
      { type: 'idle' },
    )
    return state.type === 'idle' ? (
      <Menu.Item
        as="button"
        className={clsx(className)}
        onClick={(e) => {
          // This keeps the menu open (only  on click through, not on keyboard)
          // As of Jun 2023, no solution for keyboard:
          // https://github.com/tailwindlabs/headlessui/discussions/1122
          e.preventDefault()
          setState({ type: 'pickType' })
        }}
      >
        <h2 className="">
          <span className="inline-block w-[25px]">
            <FontAwesomeIcon icon={faLink} />
          </span>{' '}
          Copy to Link
        </h2>
      </Menu.Item>
    ) : (
      <div className="px-2 mx-2 py-2.5 rounded-lg bg-gray-100">
        <button
          className="font-semibold"
          onClick={() => setState({ type: 'idle' })}
        >
          <span className="inline-block w-[25px]">
            <FontAwesomeIcon icon={faLink} />
          </span>{' '}
          Copy to Link
        </button>
        <div className="my-4">
          <_Link closeMenu={closeMenu} shortOrLong="short" />
          <_Link closeMenu={closeMenu} shortOrLong="long" />
        </div>
        <h2 className="text-xs lighten text-right">Does not include history</h2>
      </div>
    )
  },
)

const _Link = React.memo(
  ({
    className: outerClassName,
    closeMenu,
    shortOrLong,
  }: {
    className?: string
    closeMenu: () => void
    shortOrLong: 'short' | 'long'
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const { planParams, currentPortfolioBalanceInfo } = useSimulation()
    const [state, setState] = useState<
      | { type: 'idle' }
      | { type: 'generatingLink' }
      | { type: 'gotLink'; link: string }
      | { type: 'copied' }
    >({ type: 'idle' })

    const [width, setWidth] = useState(0)

    const outerDivRef = useRef<HTMLDivElement>(null)
    useEffect(() => {
      const observer = new ResizeObserver(() =>
        setWidth(fGet(outerDivRef.current).clientWidth),
      )
      observer.observe(fGet(outerDivRef.current))
      return () => observer.disconnect()
    }, [])

    const params = useMemo(() => {
      const clone = cloneJSON(planParams)
      clone.wealth.portfolioBalance = {
        updatedHere: true,
        amount: CurrentPortfolioBalance.get(currentPortfolioBalanceInfo),
      }
      return clone
    }, [currentPortfolioBalanceInfo, planParams])

    const [commitGetShortLink, isGetShortLinkRunning] =
      useMutation<PlanMenuActionCopyToLinkShortLinkMutation>(graphql`
        mutation PlanMenuActionCopyToLinkShortLinkMutation(
          $input: CreateLinkBasedPlanInput!
        ) {
          createLinkBasedPlan(input: $input) {
            id
          }
        }
      `)

    const handleGetShortLink = () => {
      commitGetShortLink({
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
    }

    const className =
      'flex w-full items-center gap-x-1 h-[35px]  border border-gray-300 rounded-md px-1  '

    return (
      <div className={clsx(outerClassName)} ref={outerDivRef}>
        <div className="" style={{ width: `${width}px` }}>
          {state.type === 'idle' ? (
            <button
              className="w-full text-start h-[35px]"
              onClick={(e) => {
                if (shortOrLong === 'short') {
                  setState({ type: 'generatingLink' })
                  handleGetShortLink()
                } else {
                  const url = appPaths.link()
                  url.searchParams.set('params', JSON.stringify(params))
                  setState({ type: 'gotLink', link: url.toString() })
                }
              }}
            >
              {shortOrLong === 'short' ? 'Short Link' : 'Long Link'}
            </button>
          ) : state.type === 'generatingLink' ? (
            <div className={className}>
              <h2 className="text-sm overflow-hidden text-ellipsis whitespace-nowrap">
                <FontAwesomeIcon
                  className="fa-spin mr-2"
                  icon={faSpinnerThird}
                />
                Generating Link...
              </h2>
            </div>
          ) : state.type === 'gotLink' ? (
            <button
              className={className}
              onClick={() => {
                setState({ type: 'copied' })
                void navigator.clipboard.writeText(state.link).then(() => {
                  window.setTimeout(() => {
                    closeMenu()
                  }, 1000)
                  return null
                })
              }}
            >
              <h2 className="text-[13px] overflow-hidden text-ellipsis whitespace-nowrap">
                {state.link}
              </h2>
              <FontAwesomeIcon className="text-lg" icon={faCopy} />
            </button>
          ) : state.type === 'copied' ? (
            <div className={className}>
              {/* <FontAwesomeIcon className="mr-2" icon={faClipboard} /> */}
              <FontAwesomeIcon className="font-bold" icon={faCheck} />
              <h2 className="text-sm">Copied to Clipboard </h2>
            </div>
          ) : (
            noCase(state)
          )}
        </div>
      </div>
    )
  },
)

const _WorkingButton = React.memo(
  ({
    className,
    children,
    onClick,
  }: {
    className?: string
    children: ReactNode
    onClick: () => void
  }) => {
    return (
      <button
        className="flex w-full items-center gap-x-1 h-[35px]  border border-gray-300 rounded-md px-1  "
        onClick={onClick}
      >
        {children}
      </button>
    )
  },
)
