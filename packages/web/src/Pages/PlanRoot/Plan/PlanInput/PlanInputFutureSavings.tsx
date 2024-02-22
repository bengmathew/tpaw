import { default as React, useRef, useState } from 'react'
import { PlanParamsExtended } from '../../../../UseSimulator/ExtentPlanParams'
import { paddingCSS } from '../../../../Utils/Geometry'
import {
    EditValueForMonthRange,
    EditValueForMonthRangeStateful,
} from '../../../Common/Inputs/EditValueForMonthRange'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
import { PlanInputSummaryValueForMonthRange } from './Helpers/PlanInputSummaryValueForMonthRange'
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
    const { planParamsExt } = useSimulation()
    const [editState, setEditState] = useState<{
      isAdd: boolean
      entryId: string
      hideInMain: boolean
    } | null>(null)

    const editRef = useRef<EditValueForMonthRangeStateful>(null)
    const { validMonthRangeAsMFN } = planParamsExt

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
                <EditValueForMonthRange
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
                  validRangeAsMFN={validMonthRangeAsMFN('future-savings')}
                  choices={{
                    start: ['now', 'numericAge', 'calendarMonth'],
                    end: [
                      'lastWorkingMonth',
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
    const { planParamsExt } = useSimulation()
    const { validMonthRangeAsMFN, months, isPersonRetired } = planParamsExt

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <p className="p-base">{`How much do you expect to save per month between now and retirement? You can enter savings from different sources separately—your savings, your partner's savings, etc.`}</p>
          <ByMonthSchedule
            className="mt-6"
            editProps={{
              defaultMonthRange: {
                type: 'startAndEnd',
                start: months.now,
                end: !isPersonRetired('person1')
                  ? months.person1.lastWorkingMonth
                  : months.person2.lastWorkingMonth,
              },
              onEdit: (entryId, isAdd) =>
                setEditState({ isAdd, entryId, hideInMain: isAdd }),
              addButtonText: 'Add',
            }}
            location="futureSavings"
            hideEntryId={
              editState && editState.hideInMain ? editState.entryId : null
            }
            allowableMonthRangeAsMFN={validMonthRangeAsMFN('future-savings')}
          />
        </div>
      </div>
    )
  },
)

export const PlanInputFutureSavingsSummary = React.memo(
  ({ planParamsExt }: { planParamsExt: PlanParamsExtended }) => {
    const { planParams } = planParamsExt
    const { validMonthRangeAsMFN } = planParamsExt
    return (
      <PlanInputSummaryValueForMonthRange
        entries={planParams.wealth.futureSavings}
        range={validMonthRangeAsMFN('future-savings')}
        planParamsExt={planParamsExt}
      />
    )
  },
)
