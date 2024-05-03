import { faCheck, faCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React from 'react'
import { appPaths } from '../../AppPaths'
import { AppPage } from '../App/AppPage'

export const PlanRootLoginOrLocal = React.memo(() => {
  return (
    <AppPage title={'Plan - TPAW Planner'}>
      <div className="h-screen flex items-center justify-center">
        <div className="max-w-[500px] w-full px-2">
          <Link
            className="block bg-gray-100 rounded-xl px-4 py-4 w-full"
            href={appPaths.login(new URL(window.location.href))}
          >
            <h2 className="text-3xl font-bold">Login</h2>
            <div className="mt-2">
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                Plan saved in your account
              </h2>
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                 Create and save multiple plans
              </h2>
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                View history of your plans
              </h2>
            </div>
          </Link>
          <Link
            className="block bg-gray-100 rounded-xl px-4 py-4 w-full mt-10 text-start"
            href={appPaths['guest']()}
            shallow={true}
          >
            <h2 className="text-3xl font-bold">Continue as Guest</h2>
            <div className="mt-2">
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                Single plan saved on the browser
              </h2>
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                Multiple plans via files or links
              </h2>
              <h2 className="flex items-center">
                <FontAwesomeIcon className="text-[6px] mr-2" icon={faCircle} />{' '}
                Plan history not available
              </h2>
            </div>
          </Link>
        </div>
      </div>
    </AppPage>
  )
})
