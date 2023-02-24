import { MonthRange } from '@tpaw/common'
import React, { useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../Utils/Geometry'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useSimulation } from '../../App/WithSimulation'
import { EditValueForMonthRange } from '../../Common/Inputs/EditValueForMonthRange'
import { usePlanContent } from '../Plan'
import {
  isPlanChartSpendingDiscretionaryType,
  isPlanChartSpendingEssentialType,
  planChartSpendingDiscretionaryTypeID,
  planChartSpendingEssentialTypeID,
} from '../PlanChart/PlanChartType'
import { useGetPlanChartURL } from '../PlanChart/UseGetPlanChartURL'
import { usePlanChartType } from '../PlanChart/UsePlanChartType'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
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
    const { months, validMonthRangeAsMFN, maxMaxAge, asMFN } = paramsExt
    const content = usePlanContent()
    const [state, setState] = useState<
      | { type: 'main' }
      | {
          type: 'edit'
          isEssential: boolean
          isAdd: boolean
          entryId: number
          hideInMain: boolean
        }
    >({ type: 'main' })

    const allowableRange = validMonthRangeAsMFN('extra-spending')
    const defaultRange: MonthRange = {
      type: 'startAndNumMonths',
      start:
        params.people.person1.ages.type === 'retired'
          ? months.now
          : months.person1.retirement,
      numMonths: Math.min(
        5 * 12,
        asMFN(maxMaxAge) + 1 - asMFN(months.person1.retirement),
      ),
    }

    const showDiscretionary =
      params.advanced.strategy !== 'SWR' ||
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
              body={content['extra-spending'].intro[params.advanced.strategy]}
              p="p-base"
            />
            {showDiscretionary && params.advanced.strategy === 'SWR' && (
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
              body={
                content['extra-spending'].essential[params.advanced.strategy]
              }
              p="p-base"
            />
            <ByMonthSchedule
              className="mt-6"
              heading={null}
              addButtonText="Add an Essential Expense"
              entries={(params) =>
                params.adjustmentsToSpending.extraSpending.essential
              }
              hideEntryId={
                state.type === 'edit' && state.isEssential && state.hideInMain
                  ? state.entryId
                  : null
              }
              allowableMonthRange={allowableRange}
              onEdit={(entryId, isAdd) =>
                setState({
                  type: 'edit',
                  isEssential: true,
                  isAdd,
                  entryId,
                  hideInMain: isAdd,
                })
              }
              defaultMonthRange={defaultRange}
            />
          </div>
          {showDiscretionary && (
            <div
              className="params-card mt-8"
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            >
              <h2 className="font-bold text-lg mb-2">Discretionary Expenses</h2>
              <Contentful.RichText
                body={
                  content['extra-spending'].discretionary[
                    params.advanced.strategy
                  ]
                }
                p="p-base"
              />
              <ByMonthSchedule
                className="mt-6"
                heading={null}
                addButtonText="Add a Discretionary Expense"
                entries={(params) =>
                  params.adjustmentsToSpending.extraSpending.discretionary
                }
                hideEntryId={
                  state.type === 'edit' &&
                  !state.isEssential &&
                  state.hideInMain
                    ? state.entryId
                    : null
                }
                allowableMonthRange={allowableRange}
                onEdit={(entryId, isAdd) =>
                  setState({
                    type: 'edit',
                    isEssential: false,
                    isAdd,
                    entryId,
                    hideInMain: isAdd,
                  })
                }
                defaultMonthRange={defaultRange}
              />
            </div>
          )}
        </div>
        {{
          input:
            state.type === 'edit'
              ? (transitionOut) => (
                  <EditValueForMonthRange
                    hasMonthRange
                    mode={state.isAdd ? 'add' : 'edit'}
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
                    getEntries={(params) =>
                      state.isEssential
                        ? params.adjustmentsToSpending.extraSpending.essential
                        : params.adjustmentsToSpending.extraSpending
                            .discretionary
                    }
                    entryId={state.entryId}
                    validRangeAsMFN={allowableRange}
                    choices={{
                      start: [
                        'now',
                        'retirement',
                        'forNumOfMonths',
                        'numericAge',
                      ],
                      end: [
                        'retirement',
                        'maxAge',
                        'numericAge',
                        'forNumOfMonths',
                      ],
                    }}
                    cardPadding={props.sizing.cardPadding}
                  />
                )
              : undefined,
        }}
      </PlanInputBody>
    )
  },
)
