import Link from 'next/link'
import React from 'react'

export const Header = React.memo(
  ({curr}: {curr: 'plan' | 'learn' | 'other'}) => {
    return (
      <div
        className={`fixed top-0  right-0 w-full sm:w-auto 
        flex justify-between sm:flex-row-reverse items-stretch gap-x-4  
         h-[47px] opacity-100 bg-gray-200 sm:bg-pageBG
       px-3  sm:px-4 sm:rounded-bl-lg border-b-2 sm:border-b-0 border-gray-700 `}
      >
        <_Button href="/" label="TPAWplanner" isCurrent={false} />
        <div className="flex gap-x-4">
          <_Button href="/plan" label="Plan" isCurrent={curr === 'plan'} />
          <_Button
            href="/learn"
            label="Learn"
            isCurrent={curr === 'learn'}
          />
        </div>
      </div>
    )
  }
)

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
      <Link href={href}>
        <a
          className={`flex items-end font-bold pb-2 ${
            isCurrent ? 'text-theme1' : ''
          }`}
        >
          {label}
        </a>
      </Link>
    )
  }
)
