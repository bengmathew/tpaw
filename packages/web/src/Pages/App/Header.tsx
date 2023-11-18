import { faBars, faHouse, faUser } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Transition } from '@headlessui/react'
import _ from 'lodash'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { useMemo, useState } from 'react'
import ReactDOM from 'react-dom'
import { appPaths } from '../../AppPaths'
import { RGB } from '../../Utils/ColorUtils'
import { mainPlanColors } from '../PlanRoot/Plan/UsePlanColors'
import { useFirebaseUser } from './WithFirebaseUser'

export const headerHeight = 47
export const Header = React.memo(({ isDark }: { isDark: boolean }) => {
  const [showMenu, setShowMenu] = useState(false)
  const firebaseUser = useFirebaseUser()
  const path = useRouter().asPath
  const curr = useMemo((): 'plan' | 'learn' | 'other' => {
    const parts = `/${path.split('/')[1]}`
    const check = (x: string) => {
      return _.isEqual(parts, x.split('/').slice(0, parts.length))
    }
    if (check(appPaths.plan().pathname)) return 'plan'
    if (check(appPaths.guest().pathname)) return 'plan'
    if (check(appPaths.learn().pathname)) return 'learn'
    return 'other'
  }, [path])

  const loginDest =
    curr === 'plan' ? new URL(window.location.href) : appPaths.plan()
  return (
    <div
      className={` print:hidden fixed top-0  right-0 
        flex justify-between  items-stretch 
          opacity-100
          rounded-bl-lg  border-gray-700 text-lg sm:text-base z-50 `}
      style={{
        height: `${headerHeight}px`,
        backgroundColor: mainPlanColors.shades.main[10].hex,
        color: mainPlanColors.shades.light[5].hex,
      }}
    >
      <Link
        className={`pl-4 pr-2 flex items-center font-bold `}
        href={appPaths.plan()}
        shallow
      >
        <FontAwesomeIcon icon={faHouse} />
      </Link>
      <div className="flex ">
        <_Button href="/learn" label="Learn" isCurrent={curr === 'learn'} />
        <button
          className="pl-2 pr-4 flex gap-x-4 items-center"
          onClick={() => setShowMenu(true)}
        >
          {firebaseUser && (
            <div
              className="w-[30px] h-[30px] rounded-full flex justify-center items-center"
              style={{
                backgroundColor: RGB.toHex(
                  RGB.addAlpha(mainPlanColors.fgForDarkBG.rgb, 0.2),
                ),
              }}
            >
              <FontAwesomeIcon icon={faUser} />
            </div>
          )}
          <FontAwesomeIcon className="text-lg" icon={faBars} />
        </button>
      </div>
      <div
        className={'absolute inset-0 pointer-events-none bg-black/60'}
        style={{
          transitionProperty: 'opacity',
          transitionDuration: '300ms',
          opacity: `${isDark ? 1 : 0}`,
        }}
      />

      {ReactDOM.createPortal(
        <Transition
          show={showMenu}
          className="fixed inset-0 page overflow-hidden"
          enter="transition-opacity duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="transition-opacity duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div
            className="absolute inset-0 bg-black bg-opacity-20"
            onClick={() => setShowMenu(false)}
          >
            <Transition.Child
              className="absolute min-w-[175px]  shadow-2xl bg-pageBG font-semibold  right-0 rounded-bl-2xl py-2 "
              enter=" transtition-transform duration-300"
              enterFrom="translate-x-[175px]"
              enterTo="translate-x-0"
              leave=" transtition-transform duration-300"
              leaveFrom="translate-x-0"
              leaveTo="translate-x-[175px]"
            >
              <div className="border-b border-gray-300 pb-2 mb-2">
                {firebaseUser ? (
                  <>
                    <h2 className="px-4 py-2">{firebaseUser.email}</h2>
                    <Link className="block px-4 py-2" href="/account">
                      Account
                    </Link>
                    <Link className="block px-4 py-2" href="/logout">
                      Logout
                    </Link>
                  </>
                ) : (
                  <Link
                    className="block px-4 py-2"
                    href={appPaths.login(loginDest)}
                  >
                    Login / Sign Up
                  </Link>
                )}
              </div>

              <Link className="block px-4 py-2" href="/about" shallow>
                About / Contact
              </Link>
              <Link className="block px-4 py-2" href="/license" shallow>
                License
              </Link>
              <Link className="block px-4 py-2" href="/privacy" shallow>
                Privacy
              </Link>
              <Link className="block px-4 py-2" href="/disclaimer" shallow>
                Disclaimer
              </Link>
            </Transition.Child>
          </div>
        </Transition>,
        window.document.body,
      )}
    </div>
  )
})

const _Button = React.memo(
  ({
    href = '',
    label,
    isCurrent,
  }: {
    href: string
    label: string
    isCurrent: boolean
  }) => {
    return (
      <Link className={`flex items-center font-bold px-2 `} href={href}>
        {label}
      </Link>
    )
  },
)
