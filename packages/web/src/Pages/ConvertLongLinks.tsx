import {
  assert,
  planParamsBackwardsCompatibleGuard,
  planParamsMigrate,
} from '@tpaw/common'
import clsx from 'clsx'
import { chain, json } from 'json-guard'
import { DateTime } from 'luxon'
import Link from 'next/link'
import React, { useState } from 'react'
import { useLazyLoadQuery } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../AppPaths'
import { errorToast } from '../Utils/CustomToasts'
import { useURLParam } from '../Utils/UseURLParam'
import { AppPage } from './App/AppPage'
import { useUserGQLArgs } from './App/WithFirebaseUser'
import { WithUser } from './App/WithUser'
import {
  fileBasedPlansLabel,
  fileBasedPlansOpenFileLabel,
} from './PlanRoot/Plan/PlanMenu/PlanMenuSection/PlanMenuSectionFile'
import {
  PLAN_FILE_EXTENSION,
  PlanFileDataFns,
} from './PlanRoot/PlanRootFile/PlanFileData'
import { useIANATimezoneName } from './PlanRoot/PlanRootHelpers/WithNonPlanParams'
import { ConvertLongLinksQuery } from './__generated__/ConvertLongLinksQuery.graphql'

export const ConvertLongLinks = React.memo(() => {
  const userGQLArgs = useUserGQLArgs()
  const data = useLazyLoadQuery<ConvertLongLinksQuery>(
    graphql`
      query ConvertLongLinksQuery($userId: ID!, $includeUser: Boolean!) {
        ...WithUser_query
      }
    `,
    { ...userGQLArgs },
  )

  return (
    <WithUser userFragmentOnQueryKey={userGQLArgs.includeUser ? data : null}>
      <_Body />
    </WithUser>
  )
})

export const _Body = React.memo(() => {
  const paramsStr = useURLParam('params')
  const { getZonedTime } = useIANATimezoneName()

  const [value, setValue] = useState(() => {
    if (!paramsStr) return ''
    const url = appPaths.link()
    url.searchParams.set('params', paramsStr)
    return url.href
  })

  const [validating, setValidating] = useState(false)
  const valid = value.trim().length > 0
  const handleConvert = () => {
    if (!valid) {
      setValidating(true)
      return
    }
    try {
      const url = new URL(value)
      const paramsStr = url.searchParams.get('params')
      assert(paramsStr)
      const planParams = chain(
        json,
        planParamsBackwardsCompatibleGuard,
      )(paramsStr).force()
      const data = PlanFileDataFns.getFromLink(planParams)
      const timestamp = planParamsMigrate(planParams).timestamp
      PlanFileDataFns.download(
        `TPAW Plan (${getZonedTime(timestamp).toLocaleString(DateTime.DATE_MED)})${PLAN_FILE_EXTENSION}`,
        data,
      )
    } catch {
      errorToast('Not a valid long link.')
    }
  }

  return (
    <AppPage
      title="Convert Long Link to File - TPAW Planner"
      className=" pt-header min-h-screen"
      style={{}}
    >
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-4 z-0">
          <div className=" ">
            <h1 className="font-bold text-4xl">Convert Long Link to File</h1>
            <div className="">
              {paramsStr && (
                <p className="mt-4 p-base">
                  You have accessed a plan using a long link.
                </p>
              )}
              <p className="mt-4 p-base">
                Long links have been replaced with files. You can convert long
                links to files by pasting them below and selecting{' '}
                {`"Convert to File"`}.
              </p>
              <p className="mt-4 p-base">
                You can open the file from{' '}
                <Link href={appPaths.plan().href} className="underline">
                  {appPaths.plan().href}
                </Link>{' '}
                by selecting {`"${fileBasedPlansOpenFileLabel}"`} from the{' '}
                {`"${fileBasedPlansLabel}"`} section of the plan menu.
              </p>
              <p className="mt-4 p-base">
                You can read more about this change{' '}
                <a
                  href="https://www.bogleheads.org/forum/viewtopic.php?p=7850328#p7850328"
                  target='_blank'
                  rel="noreferrer"
                  className="underline"
                >
                  here
                </a>
                .
              </p>

              <h2 className="font-bold text-start text-lg mt-8">
                Paste a Long Link Here
              </h2>
              <input
                type="text"
                className={clsx(
                  'px-4 py-4 rounded-lg bg-gray-100 w-full mt-4 border',
                  validating && !valid ? 'border-red-500' : 'border-gray-300',
                )}
                value={value}
                onChange={(e) => setValue(e.target.value)}
              />
              <div className="flex justify-end mt-6" onClick={handleConvert}>
                <button className={' btn2-md btn2-dark'} onClick={() => {}}>
                  Convert to File
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppPage>
  )
})
