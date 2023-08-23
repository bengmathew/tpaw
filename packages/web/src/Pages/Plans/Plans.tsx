import { faPlus } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, block, fGet, letIn } from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import Link from 'next/link'
import React, { useMemo, useState } from 'react'
import { useLazyLoadQuery } from 'react-relay'
import { graphql } from 'relay-runtime'
import { appPaths } from '../../AppPaths'
import { AppPage } from '../App/AppPage'
import { useUserGQLArgs } from '../App/WithFirebaseUser'
import { User, WithUser, useUser } from '../App/WithUser'
import { PlanInputBodyHeaderDoneButton } from '../PlanRoot/Plan/PlanInput/PlanInputBody/PlanInputBodyHeader'
import { PlanMenuActionModalCreatePlan } from '../PlanRoot/Plan/PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalCreatePlan'
import { PlansActions } from './PlansActions'
import { PlansQuery } from './__generated__/PlansQuery.graphql'

export const Plans = React.memo(() => {
  const userGQLArgs = useUserGQLArgs()

  const data = useLazyLoadQuery<PlansQuery>(
    graphql`
      query PlansQuery($userId: ID!, $includeUser: Boolean!) {
        ...WithUser_query
      }
    `,
    { ...userGQLArgs },
  )
  return (
    <WithUser userFragmentOnQueryKey={data}>
      <_Plans />
    </WithUser>
  )
})

const _Plans = React.memo(() => {
  const user = fGet(useUser())
  const [showCreateModal, setShowCreateModal] = useState(false)

  const { mainPlan, altPlans } = useMemo(() => {
    const [mainPlans, altPlans] = _.partition(user.plans, (plan) => plan.isMain)
    assert(mainPlans.length === 1)
    altPlans.sort((a, b) => b.sortTime - a.sortTime)
    return { mainPlan: mainPlans[0], altPlans }
  }, [user.plans])

  const doneURL = getPlansOnDoneURL()

  return (
    <AppPage className=" pt-header min-h-screen" title="Plans - TPAW Planner">
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-4 z-0">
          <div className="sticky top-0 inline-flex gap-x-4 z-10 bg-pageBG pt-2 pb-4 pr-4 rounded-br-lg">
            {doneURL && (
              <PlanInputBodyHeaderDoneButton className="" url={doneURL} />
            )}
            <h2 className="text-4xl font-bold ">Plans</h2>
          </div>
          <h2 className="text-2xl font-bold mt-4">Main Plan</h2>

          <_Item className="mt-4" plan={mainPlan} />

          <h2 className="text-2xl font-bold mt-10">Other Plans</h2>
          {altPlans.map((plan) => (
            <_Item key={plan.id} className=" mt-4" plan={plan} />
          ))}
          <div className="sticky bottom-1 mt-10 flex justify-end">
            <div className="flex gap-x-4 justify-end pb-4">
              <button
                className="btn2-dark btn2-lg flex items-center"
                onClick={() => setShowCreateModal(true)}
              >
                <FontAwesomeIcon className="mr-2" icon={faPlus} />
                New Plan
              </button>
            </div>
          </div>
        </div>
      </div>
      <PlanMenuActionModalCreatePlan
        show={showCreateModal}
        onHide={() => setShowCreateModal(false)}
        switchOnCreate={false}
      />
    </AppPage>
  )
})

const _Item = React.memo(
  ({
    className = '',
    plan,
  }: {
    className?: string
    plan: User['plans'][number]
  }) => {
    const href = block(() => {
      const href = appPaths['alt-plan']()
      href.searchParams.set('plan', plan.slug)
      return href
    })
    return (
      // z-0 to create a stacking context so z index of actions don't leak out.
      <div className={clsx(className, 'relative z-0 bg-gray-100 rounded-xl ')}>
        <Link
          className={`relative px-4 py-4  block`}
          href={plan.isMain ? appPaths.plan() : href}
          shallow
        >
          <h2 className={clsx('text-lg')}>{plan.label ?? '<Untitled>'}</h2>

          <h2 className="text-sm">
            Created:{' '}
            {DateTime.fromMillis(plan.addedToServerAt).toLocaleString(
              DateTime.DATE_FULL,
            )}
          </h2>
          <h2 className="text-sm">
            Last Updated:{' '}
            {DateTime.fromMillis(plan.lastSyncAt).toLocaleString(
              DateTime.DATETIME_MED,
            )}
          </h2>
        </Link>
        <PlansActions
          className="absolute top-0 right-0 z-10 py-1 px-4"
          plan={plan}
        />
      </div>
    )
  },
)

export const setPlansOnDoneURL = () =>
  window.localStorage.setItem('Plans_OnDoneURL', window.location.href)

const getPlansOnDoneURL = () =>
  letIn(window.localStorage.getItem('Plans_OnDoneURL'), (str) => {
    window.localStorage.removeItem('Plans_OnDoneURL')
    return str ? new URL(str) : null
  })
