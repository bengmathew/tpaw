import _ from 'lodash'
import { Dispatch, default as React, useEffect, useRef, useState } from 'react'
import { paddingCSS } from '../../../Utils/Geometry'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useSimulation } from '../../App/WithSimulation'
import {
  EditValueForMonthRange,
  EditValueForMonthRangeStateful,
} from '../../Common/Inputs/EditValueForMonthRange'
import { ConfirmAlert } from '../../Common/Modal/ConfirmAlert'
import { useGetSectionURL } from '../Plan'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _State =
  | { type: 'main' }
  | { type: 'edit'; isAdd: boolean; entryId: number; hideInMain: boolean }

export const PlanInputFutureSavings = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { paramsExt, params } = useSimulation()
    const [state, setState] = useState<_State>({ type: 'main' })
    const editRef = useRef<EditValueForMonthRangeStateful>(null)
    const { validMonthRangeAsMFN, allowFutureSavingsEntries } = paramsExt
    const show =
      params.plan.wealth.futureSavings.length > 0 || allowFutureSavingsEntries

    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (!show) urlUpdater.replace(summarySectionURL)
    }, [summarySectionURL, urlUpdater, show])
    if (!show) return <></>

    return (
      <PlanInputBody
        {...props}
        onBackgroundClick={() => editRef.current?.closeSections()}
      >
        {!allowFutureSavingsEntries ? (
          <_RetiredFutureSavingsCard props={props} />
        ) : (
          <_FutureSavingsCard props={props} state={state} setState={setState} />
        )}
        {{
          input:
            state.type === 'edit'
              ? (transitionOut) => (
                  <EditValueForMonthRange
                    ref={editRef}
                    hasMonthRange
                    mode={state.isAdd ? 'add' : 'edit'}
                    title={state.isAdd ? 'Add Savings' : 'Edit Savings'}
                    labelPlaceholder="E.g. From My Salary"
                    setHideInMain={(hideInMain) =>
                      setState({ ...state, hideInMain })
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({ type: 'main' })}
                    getEntries={(params) => params.wealth.futureSavings}
                    entryId={state.entryId}
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

const _RetiredFutureSavingsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { paramsExt, setPlanParams, params } = useSimulation()
    const { validMonthRangeAsMFN } = paramsExt
    const [confirm, setConfirm] = useState(false)

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <p className="p-base">{`The future savings section is no longer applicable since ${
            params.plan.people.withPartner ? 'you and your partner' : 'you'
          } are retired.`}</p>
          <button
            className="btn-sm btn-dark mt-4"
            onClick={() => {
              setConfirm(true)
            }}
          >
            Clear All Entires
          </button>

          <ByMonthSchedule
            className="mt-2"
            heading={null}
            editProps={null}
            entries={(params) => params.wealth.futureSavings}
            hideEntryId={null}
            allowableMonthRangeAsMFN={validMonthRangeAsMFN('future-savings')}
          />
        </div>
        {confirm && (
          <ConfirmAlert
            option1={{
              onClose: () => {
                setPlanParams((plan) => {
                  const clone = _.cloneDeep(plan)
                  clone.wealth.futureSavings = []
                  return clone
                })
              },
              label: 'Confirm',
            }}
            onCancel={() => setConfirm(false)}
          >
            This will clear all entries in this section.
          </ConfirmAlert>
        )}
      </div>
    )
  },
)

const _FutureSavingsCard = React.memo(
  ({
    className = '',
    props,
    state,
    setState,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    state: _State
    setState: Dispatch<_State>
  }) => {
    const { paramsExt } = useSimulation()
    const { validMonthRangeAsMFN, months, isPersonRetired } = paramsExt

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <p className="p-base">{`How much do you expect to save per month between now and retirement? You can enter savings from different sources separately—your savings, your partner's savings, etc.`}</p>
          <ByMonthSchedule
            className="mt-6"
            heading={null}
            editProps={{
              defaultMonthRange: {
                type: 'startAndEnd',
                start: months.now,
                end: !isPersonRetired('person1')
                  ? months.person1.lastWorkingMonth
                  : months.person2.lastWorkingMonth,
              },
              onEdit: (entryId, isAdd) =>
                setState({ type: 'edit', isAdd, entryId, hideInMain: isAdd }),
              addButtonText: 'Add',
            }}
            entries={(params) => params.wealth.futureSavings}
            hideEntryId={
              state.type === 'edit' && state.hideInMain ? state.entryId : null
            }
            allowableMonthRangeAsMFN={validMonthRangeAsMFN('future-savings')}
          />
        </div>
      </div>
    )
  },
)
