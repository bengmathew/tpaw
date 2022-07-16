import {default as React, useState} from 'react'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {Contentful} from '../../../Utils/Contentful'
import { paddingCSS } from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './Helpers/ByYearSchedule'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputFutureSavings = React.memo(
  ({onBack, ...props}: {onBack: () => void} & ParamsInputBodyPassThruProps) => {
    const {params} = useSimulation()
    const {validYearRange, years} = extendTPAWParams(params)
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
            body={content['future-savings'].intro[params.strategy]}
            p="p-base"
          />
          <ByYearSchedule
            className="mt-2"
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
              start: years.now,
              end:
                params.people.person1.ages.type === 'notRetired'
                  ? years.person1.lastWorkingYear
                  : years.person2.lastWorkingYear,
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
