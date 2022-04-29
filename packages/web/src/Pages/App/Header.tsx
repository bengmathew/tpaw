import Link from 'next/link'
import React from 'react'

export const headerHeight = 47
export const Header = React.memo(
  ({curr}: {curr: 'plan' | 'learn' | 'other'}) => {
    return (
      <div
        className={`fixed top-0  right-0 w-full sm:w-auto 
        flex justify-between  items-stretch gap-x-4  
          opacity-100 bg-theme1
       px-3  sm:px-4 sm:rounded-bl-lg  border-gray-700 text-lg sm:text-base z-50`}
       style={{height:`${headerHeight}px`}}
       >
        
       <_Button href="/" label="TPAWplanner" isCurrent={false} />
        <div className="flex gap-x-4">
          <_Button
            href="/learn"
            label="Learn"
            isCurrent={curr === 'learn'}
          />
          <_Button href="/plan" label="Plan" isCurrent={curr === 'plan'} />
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
          className={`flex items-center font-bold  ${
            isCurrent ? 'text-gray-100' : 'text-stone-900'
          }`}
        >
          {label}
        </a>
      </Link>
    )
  }
)
