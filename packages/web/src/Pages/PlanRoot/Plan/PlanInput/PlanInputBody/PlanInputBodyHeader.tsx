import { faLeftLong } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import Link from 'next/link'
import React, { useRef } from 'react'
import { useGetSectionURL } from '../../Plan'
import { PlanInputType } from '../Helpers/PlanInputType'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import { PlanSectionName } from '../Helpers/PlanSectionName'

export const PlanInputBodyHeader = React.memo(
  ({
    type,
    className = '',
    onBackgroundClick,
  }: {
    type: PlanInputType | 'help'
    className?: string
    onBackgroundClick?: () => void
  }) => {
    const outerDivRef = useRef<HTMLDivElement>(null)
    const getSectionURL = useGetSectionURL()

    const url = getSectionURL(
      type !== 'help'
        ? ('summary' as const)
        : getPlanInputBodyHeaderOnDoneSection(),
    )
    return (
      <div
        ref={outerDivRef}
        className={`${className} sticky top-0 flex justify-start `}
        onClick={(e) => {
          if (e.target === outerDivRef.current) onBackgroundClick?.()
        }}
      >
        <div className=" flex  items-center gap-x-4 pr-4 py-4 bg-planBG rounded-br-xl">
          <PlanInputBodyHeaderDoneButton className="" url={url} />
          <h2 className="text-xl sm:text-2xl font-bold text-start">
            {planSectionLabel(type)}
          </h2>
        </div>
      </div>
    )
  },
)

export const PlanInputBodyHeaderDoneButton = React.memo(
  ({ className, url }: { className?: string; url: URL }) => {
    return (
      <Link
        className={clsx(
          className,
          'flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5',
        )}
        href={url}
        shallow
      >
        <FontAwesomeIcon className="" icon={faLeftLong} />
        Done
      </Link>
    )
  },
)

export const setPlanInputBodyHeaderOnDoneSection = (
  section: PlanSectionName,
) => {
  window.localStorage.setItem('PlanInputBodyHeaderOnDoneSection', section)
}
const getPlanInputBodyHeaderOnDoneSection = (): PlanSectionName => {
  const result =
    (window.localStorage.getItem(
      'PlanInputBodyHeaderOnDoneSection',
    ) as PlanSectionName) ?? ('summary' as PlanSectionName)
  window.localStorage.removeItem('PlanInputBodyHeaderOnDoneSection')
  return result
}
