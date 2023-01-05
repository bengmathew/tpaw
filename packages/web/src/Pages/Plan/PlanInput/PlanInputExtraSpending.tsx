import { YearRange } from '@tpaw/common'
import React, { useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../Utils/Geometry'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useSimulation } from '../../App/WithSimulation'
import { EditValueForYearRange } from '../../Common/Inputs/EditValueForYearRange'
import { usePlanContent } from '../Plan'
import {
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
  planChartSpendingDiscretionaryTypeID,
  planChartSpendingEssentialTypeID,
} from '../PlanChart/PlanChartType'
import { useGetPlanChartURL } from '../PlanChart/UseGetPlanChartURL'
import { usePlanChartType } from '../PlanChart/UsePlanChartType'
import { ByYearSchedule } from './Helpers/ByYearSchedule'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputExtraSpending = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { paramsExt, params } = useSimulation()
    const getPlanChartURL = useGetPlanChartURL()
    const chartType = usePlanChartType()
    const urlUpdater = useURLUpdater()
    const { years, validYearRange, maxMaxAge, asYFN } = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | { type: 'main' }
      | {
          type: 'edit'
          isEssential: boolean
          isAdd: boolean
          index: number
          hideInMain: boolean
        }
    >({ type: 'main' })

    const allowableRange = validYearRange('extra-spending')
    const defaultRange: YearRange = {
      type: 'startAndNumYears',
      start:
        params.people.person1.ages.type === 'retired'
          ? years.now
          : years.person1.retirement,
      numYears: Math.min(
        5,
        asYFN(maxMaxAge) + 1 - asYFN(years.person1.retirement),
      ),
    }

    const showDiscretionary =
      params.strategy !== 'SWR' ||
      params.adjustmentsToSpending.extraSpending.discretionary.length > 0

    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className=""
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, { scale: 0.5 }),
            }}
          >
            <Contentful.RichText
              body={content['extra-spending'].intro[params.strategy]}
              p="p-base"
            />
            {showDiscretionary && params.strategy === 'SWR' && (
              <div className="p-base mt-2">
                <span className="bg-gray-300 px-2 rounded-lg ">Note</span> You
                have selected the SWR strategy. This strategy treats essential
                and discretionary expenses the same.
              </div>
            )}
          </div>
          <div
            className="params-card mt-8"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          >
            {showDiscretionary && (
              <h2 className="font-bold text-lg mb-2">Essential Expenses</h2>
            )}
            <Contentful.RichText
              body={content['extra-spending'].essential[params.strategy]}
              p="p-base"
            />
            <ByYearSchedule
              className="mt-6"
              heading={null}
              addButtonText="Add an Essential Expense"
              entries={(params) =>
                params.adjustmentsToSpending.extraSpending.essential
              }
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
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            >
              <h2 className="font-bold text-lg mb-2">Discretionary Expenses</h2>
              <Contentful.RichText
                body={content['extra-spending'].discretionary[params.strategy]}
                p="p-base"
              />
              <ByYearSchedule
                className="mt-6"
                heading={null}
                addButtonText="Add a Discretionary Expense"
                entries={(params) =>
                  params.adjustmentsToSpending.extraSpending.discretionary
                }
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
              ? (transitionOut) => (
                  <EditValueForYearRange
                    title={
                      state.isAdd
                        ? `Add ${
                            state.isEssential
                              ? 'an Essential'
                              : 'a Discretionary'
                          } Expense`
                        : `Edit ${
                            state.isEssential ? 'Essential' : 'Discretionary'
                          } Expense `
                    }
                    labelPlaceholder={
                      state.isEssential
                        ? 'E.g. Mortgage Payments'
                        : 'E.g. Travel'
                    }
                    setHideInMain={(hideInMain) =>
                      setState({ ...state, hideInMain })
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({ type: 'main' })}
                    onBeforeDelete={(id) => {
                      if (state.isEssential) {
                        if (isPlanChartSpendingEssentialType(chartType)) {
                          const currChartID =
                            planChartSpendingEssentialTypeID(chartType)
                          if (id === currChartID)
                            urlUpdater.replace(
                              getPlanChartURL('spending-total'),
                            )
                        }
                      } else {
                        if (isPlanChartSpendingDiscretionaryType(chartType)) {
                          const currChartID =
                            planChartSpendingDiscretionaryTypeID(chartType)
                          if (id === currChartID)
                            urlUpdater.replace(
                              getPlanChartURL('spending-total'),
                            )
                        }
                      }
                    }}
                    entries={(params) =>
                      state.isEssential
                        ? params.adjustmentsToSpending.extraSpending.essential
                        : params.adjustmentsToSpending.extraSpending
                            .discretionary
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
      </PlanInputBody>
    )
  },
)
