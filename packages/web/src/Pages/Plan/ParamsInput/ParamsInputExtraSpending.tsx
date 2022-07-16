import React, { useState } from 'react'
import { YearRange } from '../../../TPAWSimulator/TPAWParams'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { EditValueForYearRange } from '../../Common/Inputs/EditValueForYearRange'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType
} from '../ChartPanel/ChartPanelType'
import { usePlanContent } from '../Plan'
import { ByYearSchedule } from './Helpers/ByYearSchedule'
import { ParamsInputBody, ParamsInputBodyPassThruProps } from './ParamsInputBody'

export const ParamsInputExtraSpending = React.memo(
  ({
    chartType,
    setChartType,
    ...props
  }: {
    chartType: ChartPanelType | 'sharpe-ratio'
    setChartType: (type: ChartPanelType | 'sharpe-ratio') => void
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

    const showDiscretionary =
      params.strategy !== 'SWR' || params.withdrawals.discretionary.length > 0

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <div
            className=""
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
            }}
          >
            <Contentful.RichText
              body={content['extra-spending'].intro[params.strategy]}
              p="p-base"
            />
            {showDiscretionary && params.strategy === 'SWR' && (
              <div className="p-base mt-2">
                <span className="bg-gray-300 px-2 rounded-lg ">Note</span> You have selected the SWR strategy. This strategy treats essential and discretionary expenses the same.
              </div>
            )}
          </div>
          <div
            className="params-card mt-8"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            {showDiscretionary && (
              <h2 className="font-bold text-lg mb-2">Essential</h2>
            )}
            <ByYearSchedule
              className=""
              heading={null}
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
          </div>
          {showDiscretionary && (
            <div
              className="params-card mt-8"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            >
              <h2 className="font-bold text-lg mb-2">Discretionary</h2>
              <ByYearSchedule
                className=""
                heading={null}
                entries={params => params.withdrawals.discretionary}
                hideEntry={
                  state.type === 'edit' &&
                  !state.isEssential &&
                  state.hideInMain
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
          )}
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
