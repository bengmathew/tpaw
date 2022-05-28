import _ from 'lodash'
import React, {useState} from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './Helpers/ByYearSchedule'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputIncomeDuringRetirement = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, paramsExt} = useSimulation()
    const {validYearRange, years} = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | {type: 'main'}
      | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}
    >({type: 'main'})

    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content['income-during-retirement'].intro.fields.body}
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
      </ParamsInputBody>
    )
  }
)
