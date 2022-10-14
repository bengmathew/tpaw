import {faLeftLong} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React from 'react'
import {useSimulation} from '../../../App/WithSimulation'
import {useGetSectionURL} from '../../Plan'
import {PlanInputType} from '../Helpers/PlanInputType'
import {planSectionLabel} from '../Helpers/PlanSectionLabel'
import {planDialogOrder} from './PlanInputBodyDialogNav'

export const PlanInputBodyHeader = React.memo(
  ({
    type,
    className = '',
  }: {
    type: PlanInputType | 'results'
    className?: string
  }) => {
    const {params} = useSimulation()
    const getSectionURL = useGetSectionURL()
    const index = planDialogOrder.indexOf(type) - 1
    return (
      <div className={`${className} flex justify-start `}>
        <div className="flex  items-center gap-x-4 pr-4 bg-planBG bg-opacity-90 rounded-br-xl">
          {params.dialogMode ? (
            <div className="bg-gray-700 flex items-center justify-center text-white rounded-full w-[35px] h-[35px] font-bold">
              {index + 1}
            </div>
          ) : (
            <Link href={getSectionURL('summary')} shallow>
              <a className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5">
                <FontAwesomeIcon className="" icon={faLeftLong} />
                Done
              </a>
            </Link>
          )}
          <h2 className="text-xl sm:text-2xl font-bold text-start">
            {planSectionLabel(type, params.strategy)}
          </h2>
        </div>
      </div>
    )
  }
)
