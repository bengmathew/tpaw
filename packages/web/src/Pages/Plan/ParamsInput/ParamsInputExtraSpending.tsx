import React, {useState} from 'react'
import {YearRange} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {EditValueForYearRange} from '../../Common/Inputs/EditValueForYearRange'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from '../ChartPanel/ChartPanelType'
import {usePlanContent} from '../Plan'
import {ByYearSchedule} from './Helpers/ByYearSchedule'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputExtraSpending = React.memo(
  ({
    chartType,
    setChartType,
    ...props
  }: {
    chartType: ChartPanelType | 'sharpe-ratio'
    setChartType: (type: ChartPanelType|'sharpe-ratio') => void
  } & ParamsInputBodyPassThruProps) => {
    const {paramsExt, params} = useSimulation()
    const {years, validYearRange, maxMaxAge, asYFN} = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | {type: 'main'}
      | {
          type: 'edit'
          isEssential: boolean
          isAdd: boolean
          index: number
          hideInMain: boolean
        }
    >({type: 'main'})

    const allowableRange = validYearRange('extra-spending')
    const defaultRange: YearRange = {
      type: 'startAndNumYears',
      start:
        params.people.person1.ages.type === 'retired'
          ? years.now
          : years.person1.retirement,
      numYears: Math.min(
        5,
        asYFN(maxMaxAge) + 1 - asYFN(years.person1.retirement)
      ),
    }

    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content['extra-spending'].intro.fields.body}
            p="p-base"
          />
          <ByYearSchedule
            className="mt-6 mb-6"
            heading={'Essential'}
            entries={params => params.withdrawals.essential}
            hideEntry={
              state.type === 'edit' && state.isEssential && state.hideInMain
                ? state.index
                : null
            }
            allowableYearRange={allowableRange}
            onEdit={(index, isAdd) =>
              setState({
                type: 'edit',
                isEssential: true,
                isAdd,
                index,
                hideInMain: isAdd,
              })
            }
            defaultYearRange={defaultRange}
          />
          <ByYearSchedule
            className=""
            heading={'Discretionary'}
            entries={params => params.withdrawals.discretionary}
            hideEntry={
              state.type === 'edit' && !state.isEssential && state.hideInMain
                ? state.index
                : null
            }
            allowableYearRange={allowableRange}
            onEdit={(index, isAdd) =>
              setState({
                type: 'edit',
                isEssential: false,
                isAdd,
                index,
                hideInMain: isAdd,
              })
            }
            defaultYearRange={defaultRange}
          />
        </div>
        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditValueForYearRange
                    title={
                      state.isAdd
                        ? `Add an ${
                            state.isEssential ? 'Essential' : 'Discretionary'
                          } Expense`
                        : `Edit ${
                            state.isEssential ? 'Essential' : 'Discretionary'
                          } Expense Entry`
                    }
                    setHideInMain={hideInMain =>
                      setState({...state, hideInMain})
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({type: 'main'})}
                    onBeforeDelete={id => {
                      if (state.isEssential) {
                        if (isChartPanelSpendingEssentialType(chartType)) {
                          const currChartID =
                            chartPanelSpendingEssentialTypeID(chartType)
                          if (id === currChartID) {
                            setChartType('spending-total')
                          }
                        }
                      } else {
                        if (isChartPanelSpendingDiscretionaryType(chartType)) {
                          const currChartID =
                            chartPanelSpendingDiscretionaryTypeID(chartType)
                          if (id === currChartID) {
                            setChartType('spending-total')
                          }
                        }
                      }
                    }}
                    entries={params =>
                      state.isEssential
                        ? params.withdrawals.essential
                        : params.withdrawals.discretionary
                    }
                    index={state.index}
                    allowableRange={allowableRange}
                    choices={{
                      start: [
                        'now',
                        'retirement',
                        'forNumOfYears',
                        'numericAge',
                      ],
                      end: [
                        'retirement',
                        'maxAge',
                        'numericAge',
                        'forNumOfYears',
                      ],
                    }}
                  />
                )
              : undefined,
        }}
      </ParamsInputBody>
    )
  }
)
