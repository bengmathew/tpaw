import React, { useRef, useState } from 'react'
import { PlanParamsNormalized } from '../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { Contentful } from '../../../../Utils/Contentful'
import { paddingCSS } from '../../../../Utils/Geometry'
import {
  LabelAmountOptMonthRangeInput,
  LabelAmountOptMonthRangeInputStateful,
} from '../../../Common/Inputs/LabelAmountTimedOrUntimedInput/LabeledAmountTimedOrUntimedInput'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulationInfo } from '../../PlanRootHelpers/WithSimulation'
import { LabeledAmountTimedListInput } from '../../../Common/Inputs/LabeledAmountTimedListInput'
import { PlanInputSummaryLabeledAmountTimedList } from './Helpers/PlanInputSummaryLabeledAmountTimedList'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'
import { CalendarDayFns } from '../../../../Utils/CalendarDayFns'

export const PlanInputIncomeDuringRetirement = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParamsNormInstant } = useSimulationInfo()
    const { ages } = planParamsNormInstant
    const content = usePlanContent()
    const [editState, setEditState] = useState<{
      isAdd: boolean
      entryId: string
      hideInMain: boolean
    } | null>(null)

    const editRef = useRef<LabelAmountOptMonthRangeInputStateful>(null)
    return (
      <PlanInputBody
        {...props}
        onBackgroundClick={() => editRef.current?.closeSections()}
      >
        <div className="">
          <div
            className="params-card"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          >
            <Contentful.RichText
              body={
                content['income-during-retirement'].intro[
                  planParamsNormInstant.advanced.strategy
                ]
              }
              p="p-base mb-4"
            />
            <LabeledAmountTimedListInput
              className="mt-6"
              editProps={{
                defaultAmountAndTiming: {
                  type: 'recurring',
                  baseAmount: 0,
                  delta: null,
                  everyXMonths: 1,
                  monthRange: {
                    type: 'startAndEnd',
                    start: ages.person1.retirement.isRetired
                      ? {
                          type: 'now',
                          monthOfEntry: planParamsNormInstant.datingInfo.isDated
                            ? {
                                isDatedPlan: true,
                                calendarMonth: CalendarDayFns.toCalendarMonth(
                                  planParamsNormInstant.datingInfo.nowAsCalendarDay,
                                ),
                              }
                            : { isDatedPlan: false },
                        }
                      : {
                          type: 'namedAge',
                          age: 'retirement',
                          person: 'person1',
                        },
                    end: { type: 'namedAge', age: 'max', person: 'person1' },
                  },
                },
                onEdit: (entryId, isAdd) =>
                  setEditState({ isAdd, entryId, hideInMain: isAdd }),
                addButtonText: 'Add',
              }}
              location="incomeDuringRetirement"
              hideEntryId={
                editState && editState.hideInMain ? editState.entryId : null
              }
            />
          </div>
        </div>
        {{
          input: editState
            ? (transitionOut) => (
                <LabelAmountOptMonthRangeInput
                  ref={editRef}
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
                  choicesPreFilter={{
                    start: ['retirement', 'numericAge', 'calendarMonth', 'now'],
                    end: ['maxAge', 'numericAge', 'calendarMonth'],
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
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    return (
      <PlanInputSummaryLabeledAmountTimedList
        entries={planParamsNorm.wealth.incomeDuringRetirement}
      />
    )
  },
)
