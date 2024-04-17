import { fGet } from '@tpaw/common'
import { default as React, useRef, useState } from 'react'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { paddingCSS } from '../../../../Utils/Geometry'
import {
  LabelAmountOptMonthRangeInput,
  LabelAmountOptMonthRangeInputStateful,
} from '../../../Common/Inputs/LabelAmountTimedOrUntimedInput/LabeledAmountTimedOrUntimedInput'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { LabeledAmountTimedListInput } from '../../../Common/Inputs/LabeledAmountTimedListInput'
import { PlanInputSummaryLabeledAmountTimedList } from './Helpers/PlanInputSummaryLabeledAmountTimedList'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _EditState = {
  isAdd: boolean
  entryId: string
  hideInMain: boolean
}
export const PlanInputFutureSavings = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParamsNorm } = useSimulation()
    const { ages } = planParamsNorm
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
        <_FutureSavingsCard
          props={props}
          editState={editState}
          setEditState={setEditState}
        />
        {{
          input: editState
            ? (transitionOut) => (
                <LabelAmountOptMonthRangeInput
                  ref={editRef}
                  hasMonthRange
                  addOrEdit={editState.isAdd ? 'add' : 'edit'}
                  title={editState.isAdd ? 'Add Savings' : 'Edit Savings'}
                  labelPlaceholder="E.g. From My Salary"
                  setHideInMain={(hideInMain) =>
                    setEditState({ ...editState, hideInMain })
                  }
                  transitionOut={transitionOut}
                  onDone={() => setEditState(null)}
                  location="futureSavings"
                  entryId={editState.entryId}
                  choicesPreFilter={{
                    start: ['now', 'numericAge', 'calendarMonth'],
                    end: ['lastWorkingMonth', 'numericAge', 'calendarMonth'],
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

const _FutureSavingsCard = React.memo(
  ({
    className = '',
    props,
    editState,
    setEditState,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    editState: _EditState | null
    setEditState: (x: _EditState | null) => void
  }) => {
    const { planParamsNorm } = useSimulation()
    const { ages } = planParamsNorm

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <p className="p-base">{`How much do you expect to save per month between now and retirement? You can enter savings from different sources separately—your savings, your partner's savings, etc.`}</p>
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
                  start: {
                    type: 'now',
                    monthOfEntry: planParamsNorm.datingInfo.isDated
                      ? {
                          isDatedPlan: true,
                          calendarMonth:
                            planParamsNorm.datingInfo.nowAsCalendarMonth,
                        }
                      : { isDatedPlan: false },
                  },
                  end: {
                    type: 'namedAge',
                    age: 'lastWorkingMonth',
                    person: !ages.person1.retirement.isRetired
                      ? 'person1'
                      : 'person2',
                  },
                },
              },
              onEdit: (entryId, isAdd) =>
                setEditState({ isAdd, entryId, hideInMain: isAdd }),
              addButtonText: 'Add',
            }}
            location="futureSavings"
            hideEntryId={
              editState && editState.hideInMain ? editState.entryId : null
            }
          />
        </div>
      </div>
    )
  },
)

export const PlanInputFutureSavingsSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    return (
      <PlanInputSummaryLabeledAmountTimedList
        entries={planParamsNorm.wealth.futureSavings}
      />
    )
  },
)
