import { faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, { useRef, useState } from 'react'
import { paddingCSS } from '../../../../Utils/Geometry'
import { joinWithCommaAnd } from '../../../../Utils/JoinWithAnd'
import { useSimulation } from '../../../App/WithSimulation'
import { analyzeMonthsInParams } from '../Helpers/AnalyzeMonthsInParams'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import { PlanInputAgePerson } from './PlanInputAgePerson'
import { PlanInputAgeWithdrawalStart } from './PlanInputAgeWithdrawalStart'

export type PlanInputAgeOpenableSection =
  | `${'person1' | 'person2'}-${'current' | 'retirement' | 'max'}`
  | 'none'
export const PlanInputAge = React.memo((props: PlanInputBodyPassThruProps) => {
  const { params, setParams, paramsExt } = useSimulation()
  const contentDivRef = useRef<HTMLDivElement>(null)
  const [openSection, setOpenSection] =
    useState<PlanInputAgeOpenableSection>('none')
  const monthAnalysis = analyzeMonthsInParams(paramsExt)
  const warnings = _.uniq(
    [
      ...monthAnalysis.valueForMonthRange.filter(
        (x) => x.boundsCheck.start !== 'ok' || x.boundsCheck.end !== 'ok',
      ),
      ...monthAnalysis.glidePath.filter((x) =>
        x.analyzed.some((x) => x.issue !== 'none'),
      ),
    ]
      .map((x) => x.section)
      .map(planSectionLabel)
      .map((x) => `"${x}"`),
  )

  return (
    <PlanInputBody {...props} onBackgroundClick={() => setOpenSection('none')}>
      <div
        ref={contentDivRef}
        className=""
        onClick={(e) => {
          if (e.target === contentDivRef.current) setOpenSection('none')
        }}
      >
        <PlanInputAgePerson
          className="params-card"
          personType="person1"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          openSection={openSection}
          setOpenSection={setOpenSection}
        />
        {params.people.withPartner ? (
          <>
            <PlanInputAgePerson
              className="mt-10 params-card"
              personType="person2"
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
              openSection={openSection}
              setOpenSection={setOpenSection}
            />
            <PlanInputAgeWithdrawalStart
              className="mt-8 params-card"
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            />
          </>
        ) : (
          <button
            className="params-card font-bold text-lg mt-10 flex items-center gap-x-2 w-full"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            onClick={() => {
              const clone = _.cloneDeep(params)
              const { person1 } = clone.people
              clone.people = {
                withPartner: true,
                person1,
                person2: { ..._.cloneDeep(person1) },
                withdrawalStart: 'person1',
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
                {`Based on the ages set here, one or more months specified in the ${joinWithCommaAnd(
                  warnings,
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
