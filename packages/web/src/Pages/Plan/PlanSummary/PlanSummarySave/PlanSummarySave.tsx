import { faCaretDown, faSpinnerThird } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { formatDistance } from 'date-fns'
import Link from 'next/link'
import React, { Suspense } from 'react'
import { FirebaseUser, useFirebaseUser } from '../../../App/WithFirebaseUser'
import { useSimulation } from '../../../App/WithSimulation'
import { BasicMenu } from '../../../Common/Modal/BasicMenu'
import { loginPath } from '../../../Login'
import { setPrintOnDoneURL } from '../../../Print/Print'
import { useUser } from '../../../QueryFragments/UserFragment'
import { PlanSummarySaveLoadFromAccount } from './PlanSummarySaveLoadFromAccount'
import { PlanSummarySaveLongLink } from './PlanSummarySaveLongLink'
import { PlanSummarySaveReset } from './PlanSummarySaveReset'
import { PlanSummarySaveShortLink } from './PlanSummarySaveShortLink'
import { PlanSummarySaveToAccount } from './PlanSummarySaveToAccount'

export const PlanSummarySave = React.memo(
  ({ className = '' }: { className?: string }) => {
    const firebaseUser = useFirebaseUser()
    return (
      <BasicMenu align="right">
        <div
          className={`${className} flex items-center gap-x-2 font-medium bg-gray-700 text-white rounded-lg px-4 py-2 mt-0.5`}
        >
          Save / Reset
          <FontAwesomeIcon icon={faCaretDown} />
        </div>
        {(closeMenu) => (
          <div className="w-screen sm:w-[250px] py-3 flex flex-col items-start">
            <h2 className=" px-4 text-base font-bold mt-2 mb-2">
              Save to Account
            </h2>
            {firebaseUser ? (
              <Suspense
                fallback={
                  <div className="w-full text-start py-2 px-4 lighten-2">
                    <span className="lighten2">Logging in</span>
                    <FontAwesomeIcon
                      className="fa-spin ml-2"
                      icon={faSpinnerThird}
                    />
                  </div>
                }
              >
                <PlanSummarySaveToAccount
                  className="w-full text-start py-2 px-4"
                  firebaseUser={firebaseUser}
                  closeMenu={closeMenu}
                />
                <PlanSummarySaveLoadFromAccount
                  className="w-full text-start py-2 px-4"
                  firebaseUser={firebaseUser}
                  closeMenu={closeMenu}
                />
                <_LastSavedTime
                  className="w-full text-right px-4 py-2 text-sm lighten-2 "
                  firebaseUser={firebaseUser}
                />
              </Suspense>
            ) : (
              <Link
                className="w-full text-start py-2 px-4"
                href={loginPath()}
                shallow
              >
                Login or Sign Up
              </Link>
            )}
            <h2 className=" px-4 text-base font-bold mt-4 mb-2">Get a Link</h2>
            <PlanSummarySaveLongLink
              className="w-full text-start py-2 px-4"
              closeMenu={closeMenu}
            />
            <PlanSummarySaveShortLink
              className="w-full text-start py-2 px-4"
              closeMenu={closeMenu}
            />
            <h2 className=" px-4 text-base font-bold mt-4 mb-2">
              Print / Save as PDF
            </h2>
            <Link
              className="w-full text-start py-2 px-4"
              href={'/plan/print'}
              onClick={() => setPrintOnDoneURL(window.location.href)}
              shallow
            >
              Generate Printable Report
            </Link>
            <h2 className=" px-4 text-base font-bold mt-4 mb-2">Reset</h2>
            <PlanSummarySaveReset
              className="w-full text-start py-2 px-4"
              closeMenu={closeMenu}
            />
          </div>
        )}
      </BasicMenu>
    )
  },
)

const _LastSavedTime = React.memo(
  ({
    className = '',
    firebaseUser,
  }: {
    className?: string
    firebaseUser: FirebaseUser
  }) => {
    const { currentTime } = useSimulation()
    const user = useUser(firebaseUser)

    if (!user.plan) return <></>
    return (
      <h2 className={`${className} `}>
        Saved {formatDistance(user.plan.modifiedAt, currentTime.valueOf())} ago
      </h2>
    )
  },
)
