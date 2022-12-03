import { faLeftLong } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React from 'react'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { useSimulation } from '../../../App/WithSimulation'
import { useGetSectionURL } from '../../Plan'
import { PlanInputType } from '../Helpers/PlanInputType'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import { PlanSectionName } from '../Helpers/PlanSectionName'
import { getPlanDialogOrder } from './PlanInputBodyDialogNav'

export const PlanInputBodyHeader = React.memo(
  ({
    type,
    className = '',
  }: {
    type: PlanInputType | 'results'
    className?: string
  }) => {
    const { params, paramsExt } = useSimulation()
    const { withdrawalsStarted } = paramsExt
    const getSectionURL = useGetSectionURL()
    const index = getPlanDialogOrder(withdrawalsStarted).indexOf(type) - 1
    const urlUpdater = useURLUpdater()
    const handleClick = () => {
      const section =
        type !== 'results'
          ? ('summary' as const)
          : getPlanInputBodyHeaderOnDoneSection()
      urlUpdater.push(getSectionURL(section))
    }
    return (
      <div className={`${className} flex justify-start `}>
        <div className="flex  items-center gap-x-4 pr-4 bg-planBG bg-opacity-90 rounded-br-xl">
          {params.dialogMode ? (
            <div className="bg-gray-700 flex items-center justify-center text-white rounded-full w-[35px] h-[35px] font-bold">
              {index + 1}
            </div>
          ) : (
            <button
              className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5"
              onClick={handleClick}
            >
              <FontAwesomeIcon className="" icon={faLeftLong} />
              Done
            </button>
          )}
          <h2 className="text-xl sm:text-2xl font-bold text-start">
            {planSectionLabel(type, params.strategy)}
          </h2>
        </div>
      </div>
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
