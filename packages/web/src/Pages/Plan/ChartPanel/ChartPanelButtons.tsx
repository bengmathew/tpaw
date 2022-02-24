import {faArrowsV, faLongArrowAltRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React from 'react'

export const ChartPanelButtons = React.memo(
  ({
    className = '',
    isPortrait,
    handleRescale,
  }: {
    className?: string
    isPortrait: boolean
    handleRescale: () => void
  }) => {
    return (
      <div
        className={`${className} grid  text-base  gap-x-2
  ${isPortrait ? ' items-center ' : 'mt-3 '}`}
        style={{
          grid: isPortrait
            ? '"rescale tasks" auto/1fr auto'
            : '"rescale tasks" auto/1fr auto',
        }}
      >
        <Link href="/tasks-for-this-year">
          <a
            className="flex items-center gap-x-1 text-lg  "
            style={{gridArea: 'tasks'}}
          >
            <h2 className="font-semibold">Tasks for this year</h2>
            <FontAwesomeIcon icon={faLongArrowAltRight} />
          </a>
        </Link>
        <div className="">
          <button
            className="flex items-center py-2   rounded-full bg-gray-700 text-gray-100  btn-sm "
            onClick={handleRescale}
            style={{gridArea: 'rescale'}}
          >
            <FontAwesomeIcon className="mr-1 text-[13px]" icon={faArrowsV} />
            <span className="">Rescale</span>
          </button>
        </div>
      </div>
    )
  }
)
