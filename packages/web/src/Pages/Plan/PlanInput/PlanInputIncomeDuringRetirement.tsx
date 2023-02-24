import _ from 'lodash'
import React, { useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { EditValueForMonthRange } from '../../Common/Inputs/EditValueForMonthRange'
import { usePlanContent } from '../Plan'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputIncomeDuringRetirement = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { params, paramsExt } = useSimulation()
    const { validMonthRangeAsMFN, months } = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | { type: 'main' }
      | { type: 'edit'; isAdd: boolean; entryId: number; hideInMain: boolean }
    >({ type: 'main' })

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
                  params.advanced.strategy
                ]
              }
              p="p-base mb-4"
            />
            <ByMonthSchedule
              className="mt-6"
              heading={null}
              addButtonText="Add"
              entries={(params) => params.wealth.retirementIncome}
              hideEntryId={
                state.type === 'edit' && state.hideInMain ? state.entryId : null
              }
              allowableMonthRange={validMonthRangeAsMFN(
                'income-during-retirement',
              )}
              onEdit={(entryId, isAdd) =>
                setState({ type: 'edit', isAdd, entryId, hideInMain: isAdd })
              }
              defaultMonthRange={{
                type: 'startAndEnd',
                start:
                  params.people.person1.ages.type === 'retired'
                    ? months.now
                    : months.person1.retirement,
                end: months.person1.max,
              }}
            />
          </div>
        </div>
        {{
          input:
            state.type === 'edit'
              ? (transitionOut) => (
                  <EditValueForMonthRange
                    hasMonthRange
                    mode={state.isAdd ? 'add' : 'edit'}
                    title={
                      state.isAdd
                        ? 'Add Retirement Income'
                        : 'Edit Retirement Income'
                    }
                    labelPlaceholder="E.g. Social Security"
                    setHideInMain={(hideInMain) =>
                      setState({ ...state, hideInMain })
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({ type: 'main' })}
                    getEntries={(params) => params.wealth.retirementIncome}
                    entryId={state.entryId}
                    validRangeAsMFN={validMonthRangeAsMFN(
                      'income-during-retirement',
                    )}
                    choices={{
                      start: _.compact([
                        'retirement',
                        'numericAge',
                        'forNumOfMonths',
                        params.people.person1.ages.type === 'retired' ||
                        (params.people.withPartner &&
                          params.people.person2.ages.type === 'retired')
                          ? 'now'
                          : undefined,
                      ]),
                      end: ['maxAge', 'numericAge', 'forNumOfMonths'],
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
