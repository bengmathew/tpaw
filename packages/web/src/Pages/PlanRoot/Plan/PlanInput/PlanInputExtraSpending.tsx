import { MonthRange } from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, { useState } from 'react'
import { Contentful } from '../../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../../Utils/Geometry'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { EditValueForMonthRange } from '../../../Common/Inputs/EditValueForMonthRange'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import {
  isPlanResultsChartSpendingDiscretionaryType,
  isPlanResultsChartSpendingEssentialType,
  planResultsChartSpendingDiscretionaryTypeID,
  planResultsChartSpendingEssentialTypeID,
} from '../PlanResults/PlanResultsChartType'
import { useGetPlanResultsChartURL } from '../PlanResults/UseGetPlanResultsChartURL'
import { usePlanResultsChartType } from '../PlanResults/UsePlanResultsChartType'
import { ByMonthSchedule } from './Helpers/ByMonthSchedule'
import { PlanInputSummaryValueForMonthRange } from './Helpers/PlanInputSummaryValueForMonthRange'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputExtraSpending = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParamsExt, planParams } = useSimulation()
    const getPlanChartURL = useGetPlanResultsChartURL()
    const chartType = usePlanResultsChartType()
    const urlUpdater = useURLUpdater()
    const { months, validMonthRangeAsMFN, maxMaxAge, asMFN, isPersonRetired } =
      planParamsExt
    const content = usePlanContent()
    const [editState, setEditState] = useState<{
      isEssential: boolean
      isAdd: boolean
      entryId: string
      hideInMain: boolean
    } | null>(null)

    const allowableRange = validMonthRangeAsMFN('extra-spending')
    const defaultRange: MonthRange = {
      type: 'startAndNumMonths',
      start: isPersonRetired('person1')
        ? months.now
        : months.person1.retirement,
      numMonths: Math.min(
        5 * 12,
        asMFN(maxMaxAge) + 1 - asMFN(months.person1.retirement),
      ),
    }

    const showDiscretionary =
      planParams.advanced.strategy !== 'SWR' ||
      _.values(planParams.adjustmentsToSpending.extraSpending.discretionary)
        .length > 0

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
              body={
                content['extra-spending'].intro[planParams.advanced.strategy]
              }
              p="p-base"
            />
            {showDiscretionary && planParams.advanced.strategy === 'SWR' && (
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
                content['extra-spending'].essential[
                  planParams.advanced.strategy
                ]
              }
              p="p-base"
            />
            <ByMonthSchedule
              className="mt-6"
              heading={null}
              editProps={{
                defaultMonthRange: defaultRange,
                onEdit: (entryId, isAdd) =>
                  setEditState({
                    isEssential: true,
                    isAdd,
                    entryId,
                    hideInMain: isAdd,
                  }),
                addButtonText: 'Add an Essential Expense',
              }}
              location="extraSpendingEssential"
              hideEntryId={
                editState && editState.isEssential && editState.hideInMain
                  ? editState.entryId
                  : null
              }
              allowableMonthRangeAsMFN={allowableRange}
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
                    planParams.advanced.strategy
                  ]
                }
                p="p-base"
              />
              <ByMonthSchedule
                className="mt-6"
                heading={null}
                editProps={{
                  defaultMonthRange: defaultRange,
                  onEdit: (entryId, isAdd) =>
                    setEditState({
                      isEssential: false,
                      isAdd,
                      entryId,
                      hideInMain: isAdd,
                    }),

                  addButtonText: 'Add a Discretionary Expense',
                }}
                location="extraSpendingDiscretionary"
                hideEntryId={
                  editState && !editState.isEssential && editState.hideInMain
                    ? editState.entryId
                    : null
                }
                allowableMonthRangeAsMFN={allowableRange}
              />
            </div>
          )}
        </div>
        {{
          input: editState
            ? (transitionOut) => (
                <EditValueForMonthRange
                  hasMonthRange
                  addOrEdit={editState.isAdd ? 'add' : 'edit'}
                  title={
                    editState.isAdd
                      ? `Add ${
                          editState.isEssential
                            ? 'an Essential'
                            : 'a Discretionary'
                        } Expense`
                      : `Edit ${
                          editState.isEssential ? 'Essential' : 'Discretionary'
                        } Expense `
                  }
                  labelPlaceholder={
                    editState.isEssential
                      ? 'E.g. Mortgage Payments'
                      : 'E.g. Travel'
                  }
                  setHideInMain={(hideInMain) =>
                    setEditState({ ...editState, hideInMain })
                  }
                  transitionOut={transitionOut}
                  onDone={() => setEditState(null)}
                  onBeforeDelete={(id) => {
                    if (editState.isEssential) {
                      if (isPlanResultsChartSpendingEssentialType(chartType)) {
                        const currChartID =
                          planResultsChartSpendingEssentialTypeID(chartType)
                        if (id === currChartID)
                          urlUpdater.replace(getPlanChartURL('spending-total'))
                      }
                    } else {
                      if (
                        isPlanResultsChartSpendingDiscretionaryType(chartType)
                      ) {
                        const currChartID =
                          planResultsChartSpendingDiscretionaryTypeID(chartType)
                        if (id === currChartID)
                          urlUpdater.replace(getPlanChartURL('spending-total'))
                      }
                    }
                  }}
                  location={
                    editState.isEssential
                      ? 'extraSpendingEssential'
                      : 'extraSpendingDiscretionary'
                  }
                  entryId={editState.entryId}
                  validRangeAsMFN={allowableRange}
                  choices={{
                    start: [
                      'now',
                      'retirement',
                      'numericAge',
                      'calendarMonth',
                      'forNumOfMonths',
                    ],
                    end: [
                      'retirement',
                      'maxAge',
                      'numericAge',
                      'calendarMonth',
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

export const PlanInputExtraSpendingSummary = React.memo(
  ({ forPrint = false }: { forPrint?: boolean }) => {
    const { planParams, planParamsExt } = useSimulation()
    const { validMonthRangeAsMFN } = planParamsExt
    const { essential, discretionary } =
      planParams.adjustmentsToSpending.extraSpending
    const showLabels = planParams.advanced.strategy !== 'SWR'
    return (
      <>
        {_.values(essential).length === 0 &&
          _.values(discretionary).length === 0 && <h2>None</h2>}
        {_.values(essential).length > 0 && (
          <>
            {showLabels && (
              <h2
                className={clsx(
                  'mt-1 font-medium ',
                  forPrint && 'text-lg mt-2',
                )}
              >
                Essential
              </h2>
            )}
            <PlanInputSummaryValueForMonthRange
              entries={essential}
              range={validMonthRangeAsMFN('extra-spending')}
            />
          </>
        )}
        {_.values(discretionary).length > 0 && (
          <>
            {showLabels && (
              <h2
                className={clsx(
                  'mt-1 font-medium ',
                  forPrint && 'text-lg mt-2',
                )}
              >
                Discretionary
              </h2>
            )}
            <PlanInputSummaryValueForMonthRange
              entries={discretionary}
              range={validMonthRangeAsMFN('extra-spending')}
            />
          </>
        )}
      </>
    )
  },
)
