import _ from 'lodash'
import React, {useState} from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './Helpers/ByYearSchedule'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputIncomeDuringRetirement = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params, paramsExt} = useSimulation()
    const {validYearRange, years} = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | {type: 'main'}
      | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}
    >({type: 'main'})

    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="params-card"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <Contentful.RichText
              body={content['income-during-retirement'].intro[params.strategy]}
              p="p-base mb-4"
            />
            <ByYearSchedule
              className="mt-6"
              heading={null}
              addButtonText="Add Retirement Income"
              entries={params => params.retirementIncome}
              hideEntry={
                state.type === 'edit' && state.hideInMain ? state.index : null
              }
              allowableYearRange={validYearRange('income-during-retirement')}
              onEdit={(index, isAdd) =>
                setState({type: 'edit', isAdd, index, hideInMain: isAdd})
              }
              defaultYearRange={{
                type: 'startAndEnd',
                start:
                  params.people.person1.ages.type === 'retired'
                    ? years.now
                    : years.person1.retirement,
                end: years.person1.max,
              }}
            />
          </div>
        </div>
        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditValueForYearRange
                    title={
                      state.isAdd
                        ? 'Add Retirement Income'
                        : 'Edit Retirement Income'
                    }
                    setHideInMain={hideInMain =>
                      setState({...state, hideInMain})
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({type: 'main'})}
                    entries={params => params.retirementIncome}
                    index={state.index}
                    allowableRange={validYearRange('income-during-retirement')}
                    choices={{
                      start: _.compact([
                        'retirement',
                        'numericAge',
                        'forNumOfYears',
                        params.people.person1.ages.type === 'retired' ||
                        (params.people.withPartner &&
                          params.people.person2.ages.type === 'retired')
                          ? 'now'
                          : undefined,
                      ]),
                      end: ['maxAge', 'numericAge', 'forNumOfYears'],
                    }}
                  />
                )
              : undefined,
        }}
      </PlanInputBody>
    )
  }
)
