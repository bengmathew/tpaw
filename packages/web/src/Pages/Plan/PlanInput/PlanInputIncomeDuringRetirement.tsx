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
    const { validMonthRangeAsMFN, months, isPersonRetired } = paramsExt
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
                  params.plan.advanced.strategy
                ]
              }
              p="p-base mb-4"
            />
            <ByMonthSchedule
              className="mt-6"
              heading={null}
              editProps={{
                defaultMonthRange: {
                  type: 'startAndEnd',
                  start: isPersonRetired('person1')
                    ? months.now
                    : months.person1.retirement,
                  end: months.person1.max,
                },
                onEdit: (entryId, isAdd) =>
                  setState({ type: 'edit', isAdd, entryId, hideInMain: isAdd }),
                addButtonText: 'Add',
              }}
              entries={(params) => params.wealth.retirementIncome}
              hideEntryId={
                state.type === 'edit' && state.hideInMain ? state.entryId : null
              }
              allowableMonthRangeAsMFN={validMonthRangeAsMFN(
                'income-during-retirement',
              )}
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
                        !isPersonRetired('person1') ||
                        (params.plan.people.withPartner &&
                          !isPersonRetired('person2'))
                          ? 'retirement'
                          : undefined,
                        'numericAge',
                        'calendarMonth',
                        'forNumOfMonths',
                        isPersonRetired('person1') ||
                        (params.plan.people.withPartner &&
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
