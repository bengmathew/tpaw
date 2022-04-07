import {default as React, useState} from 'react'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {Contentful} from '../../../Utils/Contentful'
import {useSimulation} from '../../App/WithSimulation'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {ParamsInputBody, ParamsInputBodyProps} from './ParamsInputBody'

export const ParamsInputFutureSavings = React.memo(
  ({onBack, ...props}: {onBack: () => void} & ParamsInputBodyProps) => {
    const {params} = useSimulation()
    const {validYearRange, years} = extendTPAWParams(params)
    const content = usePlanContent()

    const [state, setState] = useState<
      | {type: 'main'}
      | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}
    >({type: 'main'})

    return (
      <ParamsInputBody {...props}>
        <div className="">
          <Contentful.RichText
            body={content.futureSavings.intro.fields.body}
            p="p-base"
          />
          <ByYearSchedule
            className=""
            heading={null}
            entries={params => params.savings}
            hideEntry={
              state.type === 'edit' && state.hideInMain ? state.index : null
            }
            allowableYearRange={validYearRange('future-savings')}
            onEdit={(index, isAdd) =>
              setState({type: 'edit', isAdd, index, hideInMain: isAdd})
            }
            defaultYearRange={{
              type: 'startAndEnd',
              start: {type: 'now'},
              end: {
                type: 'namedAge',
                person: 'person1',
                age: 'lastWorkingYear',
              },
            }}
          />
        </div>
        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditValueForYearRange
                    title={
                      state.isAdd ? 'Add to Savings' : 'Edit Savings Entry'
                    }
                    setHideInMain={hideInMain =>
                      setState({...state, hideInMain})
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({type: 'main'})}
                    entries={params => params.savings}
                    index={state.index}
                    allowableRange={validYearRange('future-savings')}
                    choices={{
                      start: ['now', 'numericAge'],
                      end: ['lastWorkingYear', 'numericAge', 'forNumOfYears'],
                    }}
                  />
                )
              : undefined,
        }}
      </ParamsInputBody>
    )
  }
)
