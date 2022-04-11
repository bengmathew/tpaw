import React, {useState} from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {ParamsInputBody, ParamsInputBodyProps} from './ParamsInputBody'

export const ParamsInputIncomeDuringRetirement = React.memo(
  (props: ParamsInputBodyProps) => {
    const {params, paramsExt} = useSimulation()
    const {validYearRange, years} = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | {type: 'main'}
      | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}
    >({type: 'main'})

    return (
      <ParamsInputBody {...props}>
        <div className="">
          <Contentful.RichText
            body={content.incomeDuringRetirement.intro.fields.body}
            p="p-base"
          />
          <ByYearSchedule
            className=""
            heading={null}
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
        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditValueForYearRange
                    title={
                      state.isAdd
                        ? 'Add to Retirement Income'
                        : 'Edit Retirement Income Entry'
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
                      start: ['retirement', 'numericAge', 'forNumOfYears'],
                      end: ['maxAge', 'numericAge', 'forNumOfYears'],
                    }}
                  />
                )
              : undefined,
        }}
      </ParamsInputBody>
    )
  }
)
