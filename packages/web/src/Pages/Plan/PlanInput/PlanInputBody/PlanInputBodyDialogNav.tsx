import {faChevronLeft, faChevronRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import Link from 'next/link'
import React from 'react'
import {useSimulation} from '../../../App/WithSimulation'
import {useGetSectionURL} from '../../Plan'
import {PlanInputType} from '../Helpers/PlanInputType'
import {PlanSectionName} from '../Helpers/PlanSectionName'

export const getPlanDialogOrder = (
  withdrawalStarted: boolean
): readonly PlanSectionName[] =>
  _.compact([
    'welcome',
    'age',
    'current-portfolio-balance',
    withdrawalStarted ?undefined: 'future-savings' ,
    'income-during-retirement',
    'results',
  ])

export const PlanInputBodyDialogNav = React.memo(
  ({className = '', type}: {className?: string; type: PlanInputType}) => {
    const {params, paramsExt} = useSimulation()
    const {withdrawalsStarted} = paramsExt
    const sectionURLFn = useGetSectionURL()
    const planDialogOrder = getPlanDialogOrder(withdrawalsStarted)

    const index = planDialogOrder.indexOf(type)
    if (!params.dialogMode || index <= 0) return <></>

    const prev = planDialogOrder[index - 1]
    const next = planDialogOrder[index + 1]
    const remaining = planDialogOrder.length - 1 - index - 1

    return (
      <div className={`${className}`}>
        <div className={`flex mt-4 items-stretch justify-between`}>
          <Link href={sectionURLFn(prev)} shallow>
            <a className="flex items-center pl-4 -ml-4 pr-6">
              <FontAwesomeIcon className="text-lg" icon={faChevronLeft} />
            </a>
          </Link>

          <Link href={sectionURLFn(next)} shallow>
            <a className="flex flex-col items-end">
              <h2 className="flex items-center gap-x-2  ">
                {remaining === 0 ? (
                  <div className="font-bold text-xl"> Results</div>
                ) : (
                  <div className="flex flex-col items-end">
                    <div className="font-bold text-xl">Next</div>
                    <div className="text-sm -mt-1 lighten">
                      {remaining} remaining
                    </div>
                  </div>
                )}
                <FontAwesomeIcon className="text-lg" icon={faChevronRight} />
              </h2>
            </a>
          </Link>
        </div>
      </div>
    )
  }
)
