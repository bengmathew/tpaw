import _ from 'lodash'
import React, { useState } from 'react'
import { PlanParamsExtended } from '../../../../UseSimulator/ExtentPlanParams'
import { Contentful } from '../../../../Utils/Contentful'
import { paddingCSS } from '../../../../Utils/Geometry'
import { EditValueForMonthRange } from '../../../Common/Inputs/EditValueForMonthRange'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
import { PlanInputSummaryValueForMonthRange } from './Helpers/PlanInputSummaryValueForMonthRange'
import {
    PlanInputBody,
    PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputIncomeDuringRetirement = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParams, planParamsExt } = useSimulation()
    const { validMonthRangeAsMFN, months, isPersonRetired } = planParamsExt
    const content = usePlanContent()
    const [editState, setEditState] = useState<{
      isAdd: boolean
      entryId: string
      hideInMain: boolean
    } | null>(null)

    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="params-card"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          >
            <Contentful.RichText
              body={
                content['income-during-retirement'].intro[
                  planParams.advanced.strategy
                ]
              }
              p="p-base mb-4"
            />
            <ByMonthSchedule
              className="mt-6"
              editProps={{
                defaultMonthRange: {
                  type: 'startAndEnd',
                  start: isPersonRetired('person1')
                    ? months.now
                    : months.person1.retirement,
                  end: months.person1.max,
                },
                onEdit: (entryId, isAdd) =>
                  setEditState({ isAdd, entryId, hideInMain: isAdd }),
                addButtonText: 'Add',
              }}
              location="incomeDuringRetirement"
              hideEntryId={
                editState && editState.hideInMain ? editState.entryId : null
              }
              allowableMonthRangeAsMFN={validMonthRangeAsMFN(
                'income-during-retirement',
              )}
            />
          </div>
        </div>
        {{
          input: editState
            ? (transitionOut) => (
                <EditValueForMonthRange
                  hasMonthRange
                  addOrEdit={editState.isAdd ? 'add' : 'edit'}
                  title={
                    editState.isAdd
                      ? 'Add Retirement Income'
                      : 'Edit Retirement Income'
                  }
                  labelPlaceholder="E.g. Social Security"
                  setHideInMain={(hideInMain) =>
                    setEditState({ ...editState, hideInMain })
                  }
                  transitionOut={transitionOut}
                  onDone={() => setEditState(null)}
                  location="incomeDuringRetirement"
                  entryId={editState.entryId}
                  validRangeAsMFN={validMonthRangeAsMFN(
                    'income-during-retirement',
                  )}
                  choices={{
                    start: _.compact([
                      !isPersonRetired('person1') ||
                      (planParams.people.withPartner &&
                        !isPersonRetired('person2'))
                        ? 'retirement'
                        : undefined,
                      'numericAge',
                      'calendarMonth',
                      'forNumOfMonths',
                      isPersonRetired('person1') ||
                      (planParams.people.withPartner &&
                        isPersonRetired('person2'))
                        ? 'now'
                        : undefined,
                    ]),
                    end: [
                      'maxAge',
                      'numericAge',
                      'calendarMonth',
                      'forNumOfMonths',
                    ],
                  }}
                  cardPadding={props.sizing.cardPadding}
                />
              )
            : undefined,
        }}
      </PlanInputBody>
    )
  },
)

export const PlanInputIncomeDuringRetirementSummary = React.memo(
  ({ planParamsExt }: { planParamsExt: PlanParamsExtended }) => {
    const { planParams } = planParamsExt
    const { validMonthRangeAsMFN } = planParamsExt
    return (
      <PlanInputSummaryValueForMonthRange
        entries={planParams.wealth.incomeDuringRetirement}
        range={validMonthRangeAsMFN('income-during-retirement')}
        planParamsExt={planParamsExt}
      />
    )
  },
)
