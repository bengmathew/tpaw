import React, {useState} from 'react'
import {YearRange} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
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
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {ParamsInputBody, ParamsInputBodyProps} from './ParamsInputBody'

export const ParamsInputExtraSpending = React.memo(
  ({
    chartType,
    setChartType,
    ...props
  }: {
    chartType: ChartPanelType
    setChartType: (type: ChartPanelType) => void
  } & ParamsInputBodyProps) => {
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
      <ParamsInputBody {...props}>
        <div className="">
          <Contentful.RichText
            body={content.extraSpending.intro.fields.body}
            p="mb-6 p-base"
          />
          <ByYearSchedule
            className="mb-6"
            heading={'Essential'}
            entries={params => params.withdrawals.fundedByBonds}
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
            entries={params => params.withdrawals.fundedByRiskPortfolio}
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
                        ? params.withdrawals.fundedByBonds
                        : params.withdrawals.fundedByRiskPortfolio
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
