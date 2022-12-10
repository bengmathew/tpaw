import {default as React, Dispatch, useEffect, useState} from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {useGetSectionURL, usePlanContent} from '../Plan'
import {ByYearSchedule} from './Helpers/ByYearSchedule'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _State =
  | {type: 'main'}
  | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}

export const PlanInputFutureSavings = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {paramsExt} = useSimulation()
    const [state, setState] = useState<_State>({type: 'main'})
    const {validYearRange, withdrawalsStarted} = paramsExt

    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (withdrawalsStarted) urlUpdater.replace(summarySectionURL)
    }, [withdrawalsStarted, summarySectionURL, urlUpdater])
    if (withdrawalsStarted) return <></>

    return (
      <PlanInputBody {...props}>
        <_FutureSavingsCard props={props} state={state} setState={setState} />
        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditValueForYearRange
                    title={state.isAdd ? 'Add Savings' : 'Edit Savings'}
                    labelPlaceholder="E.g. From My Salary"
                    setHideInMain={hideInMain =>
                      setState({...state, hideInMain})
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({type: 'main'})}
                    entries={params => params.futureSavings}
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
      </PlanInputBody>
    )
  }
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
    const {params, paramsExt} = useSimulation()
    const {validYearRange, years} = paramsExt

    const content = usePlanContent()

    return (
      <div className={`${className}`}>
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content['future-savings'].intro[params.strategy]}
            p="p-base"
          />
          <ByYearSchedule
            className="mt-6"
            heading={null}
            addButtonText="Add Savings"
            entries={params => params.futureSavings}
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
      </div>
    )
  }
)
