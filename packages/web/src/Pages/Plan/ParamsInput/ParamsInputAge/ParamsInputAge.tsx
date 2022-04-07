import { faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import { joinWithCommaAnd } from '../../../../Utils/JoinWithAnd'
import { useSimulation } from '../../../App/WithSimulation'
import { mapAllValueForYearsRangesInParams } from '../Helpers/MapAllValueForYearRangesInParams'
import { paramsInputValidateYearRange } from '../Helpers/ParamInputValidate'
import { paramsInputLabel } from '../Helpers/ParamsInputLabel'
import { ParamsInputBody, ParamsInputBodyProps } from '../ParamsInputBody'
import { ParamsInputAgePerson } from './ParamsInputAgePerson'
import { ParamsInputAgeWithdrawalStart } from './ParamsInputAgeWithdrawalStart'

export const ParamsInputAge = React.memo((props: ParamsInputBodyProps) => {
  const {params, setParams} = useSimulation()
  const warnings = _.uniq(
    mapAllValueForYearsRangesInParams(params, paramsInputValidateYearRange)
      .filter(x => x.boundsCheck.start !== 'ok' || x.boundsCheck.end !== 'ok')
      .map(x => x.sectionType)
      .map(paramsInputLabel)
      .map(x => `"${x}"`)
  )

  return (
    <ParamsInputBody {...props}>
      <div className="">
        <ParamsInputAgePerson className="" type="person1" />
        {params.people.withPartner ? (
          <>
            <ParamsInputAgePerson className="mt-10 " type="person2" />
            <div className="">
              <h2 className="mt-10 font-bold text-lg">Age Used in Graphs</h2>
              <p className="mt-3">
                All graphs are displayed in terms of{' '}
                {params.people.xAxis === 'person1' ? 'your' : "your partner's"}{' '}
                age.
              </p>
            </div>
            <ParamsInputAgeWithdrawalStart className="mt-8" />
          </>
        ) : (
          <button
            className="font-bold text-lg mt-10 flex items-center gap-x-2"
            onClick={() => {
              const clone = _.cloneDeep(params)
              const {person1} = clone.people
              clone.people = {
                withPartner: true,
                person1,
                person2: {..._.cloneDeep(person1), displayName: null},
                withdrawalStart: 'person1',
                xAxis: 'person1',
              }
              setParams(clone)
            }}
          >
            <h2 className="">Add a Partner</h2>
            <FontAwesomeIcon className=" " icon={faPlus} />
          </button>
        )}
      </div>
      {{
        error:
          warnings.length === 0 ? undefined : (
            <div className="p-base">
              <h2 className="text-errorFG ">
                <FontAwesomeIcon
                  className="mr-2 text-errorFG"
                  icon={faExclamationCircle}
                />{' '}
                {`Based on the ages set here, one or more entries in the ${joinWithCommaAnd(
                  warnings
                )} ${
                  warnings.length > 1 ? 'sections' : 'section'
                } has a year that is out of range or an end year that is earlier than the start year.`}
              </h2>
            </div>
          ),
      }}
    </ParamsInputBody>
  )
})
