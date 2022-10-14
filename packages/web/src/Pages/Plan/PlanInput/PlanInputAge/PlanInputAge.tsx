import {faPlus} from '@fortawesome/pro-regular-svg-icons'
import {faExclamationCircle} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {paddingCSS} from '../../../../Utils/Geometry'
import {joinWithCommaAnd} from '../../../../Utils/JoinWithAnd'
import {useSimulation} from '../../../App/WithSimulation'
import {analyzeYearsInParams} from '../Helpers/AnalyzeYearsInParams'
import {planSectionLabel} from '../Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {PlanInputAgePerson} from './PlanInputAgePerson'
import {PlanInputAgeWithdrawalStart} from './PlanInputAgeWithdrawalStart'

export const PlanInputAge = React.memo((props: PlanInputBodyPassThruProps) => {
  const {params, setParams, paramsExt} = useSimulation()
  const yearAnalysis = analyzeYearsInParams(paramsExt)
  const warnings = _.uniq(
    [
      ...yearAnalysis.valueForYearRange.filter(
        x => x.boundsCheck.start !== 'ok' || x.boundsCheck.end !== 'ok'
      ),
      ...yearAnalysis.glidePath.filter(x =>
        x.analyzed.some(x => x.issue !== 'none')
      ),
    ]
      .map(x => x.section)
      .map(planSectionLabel)
      .map(x => `"${x}"`)
  )

  return (
    <PlanInputBody {...props} >
      <div className="">
        <PlanInputAgePerson
          className="params-card"
          type="person1"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        />
        {params.people.withPartner ? (
          <>
            <PlanInputAgePerson
              className="mt-10 params-card"
              type="person2"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            />
            <div
              className="mt-10  params-card"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            >
              <h2 className="font-bold text-lg">Age Used in Graphs</h2>
              <p className="mt-3">
                All graphs are displayed in terms of{' '}
                {params.people.xAxis === 'person1' ? 'your' : "your partner's"}{' '}
                age.
              </p>
            </div>
            <PlanInputAgeWithdrawalStart
              className="mt-8 params-card"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            />
          </>
        ) : (
          <button
            className="params-card font-bold text-lg mt-10 flex items-center gap-x-2 w-full"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
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
                {`Based on the ages set here, one or more years specified in the ${joinWithCommaAnd(
                  warnings
                )} ${
                  warnings.length > 1 ? 'sections' : 'section'
                } needs to be updated.`}
              </h2>
            </div>
          ),
      }}
    </PlanInputBody>
  )
})
