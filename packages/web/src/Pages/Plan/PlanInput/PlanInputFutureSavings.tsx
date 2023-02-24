import { default as React, Dispatch, useEffect, useRef, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS } from '../../../Utils/Geometry'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useSimulation } from '../../App/WithSimulation'
import {
  EditValueForMonthRange,
  EditValueForMonthRangeStateful,
} from '../../Common/Inputs/EditValueForMonthRange'
import { useGetSectionURL, usePlanContent } from '../Plan'
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
    const { paramsExt } = useSimulation()
    const [state, setState] = useState<_State>({ type: 'main' })
    const editRef = useRef<EditValueForMonthRangeStateful>(null)
    const { validMonthRangeAsMFN, withdrawalsStarted } = paramsExt

    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (withdrawalsStarted) urlUpdater.replace(summarySectionURL)
    }, [withdrawalsStarted, summarySectionURL, urlUpdater])
    if (withdrawalsStarted) return <></>

    return (
      <PlanInputBody
        {...props}
        onBackgroundClick={() => editRef.current?.closeSections()}
      >
        <_FutureSavingsCard props={props} state={state} setState={setState} />
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
                      start: ['now', 'numericAge'],
                      end: ['lastWorkingMonth', 'numericAge', 'forNumOfMonths'],
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
    state,
    setState,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    state: _State
    setState: Dispatch<_State>
  }) => {
    const { params, paramsExt } = useSimulation()
    const { validMonthRangeAsMFN, months } = paramsExt

    const content = usePlanContent()

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
        >
          <Contentful.RichText
            body={content['future-savings'].intro[params.advanced.strategy]}
            p="p-base"
          />
          <ByMonthSchedule
            className="mt-6"
            heading={null}
            addButtonText="Add"
            entries={(params) => params.wealth.futureSavings}
            hideEntryId={
              state.type === 'edit' && state.hideInMain ? state.entryId : null
            }
            allowableMonthRange={validMonthRangeAsMFN('future-savings')}
            onEdit={(entryId, isAdd) =>
              setState({ type: 'edit', isAdd, entryId, hideInMain: isAdd })
            }
            defaultMonthRange={{
              type: 'startAndEnd',
              start: months.now,
              end:
                params.people.person1.ages.type === 'notRetired'
                  ? months.person1.lastWorkingMonth
                  : months.person2.lastWorkingMonth,
            }}
          />
        </div>
      </div>
    )
  },
)
