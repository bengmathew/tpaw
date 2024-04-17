import { LabeledAmountTimed } from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, { useRef, useState } from 'react'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { Contentful } from '../../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../../Utils/Geometry'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import {
  LabelAmountOptMonthRangeInput,
  LabelAmountOptMonthRangeInputStateful,
} from '../../../Common/Inputs/LabelAmountTimedOrUntimedInput/LabeledAmountTimedOrUntimedInput'
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
import { LabeledAmountTimedListInput } from '../../../Common/Inputs/LabeledAmountTimedListInput'
import { PlanInputSummaryLabeledAmountTimedList } from './Helpers/PlanInputSummaryLabeledAmountTimedList'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputExtraSpending = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParamsNorm } = useSimulation()
    const { ages } = planParamsNorm
    const getPlanChartURL = useGetPlanResultsChartURL()
    const chartType = usePlanResultsChartType()
    const urlUpdater = useURLUpdater()
    const content = usePlanContent()
    const [editState, setEditState] = useState<{
      isEssential: boolean
      isAdd: boolean
      entryId: string
      hideInMain: boolean
    } | null>(null)

    const defaultAmountAndTiming: LabeledAmountTimed['amountAndTiming'] = {
      type: 'recurring',
      baseAmount: 0,
      delta: null,
      everyXMonths: 1,
      monthRange: {
        type: 'startAndDuration',
        start: ages.person1.retirement.isRetired
          ? {
              type: 'now',
              monthOfEntry: planParamsNorm.datingInfo.isDated
                ? {
                    isDatedPlan: true,
                    calendarMonth: planParamsNorm.datingInfo.nowAsCalendarMonth,
                  }
                : { isDatedPlan: false },
            }
          : { type: 'namedAge', age: 'retirement', person: 'person1' },
        duration: {
          inMonths: Math.min(5 * 12, ages.person1.retirement.numMonthsLeft),
        },
      },
    }

    const showDiscretionary =
      planParamsNorm.advanced.strategy !== 'SWR' ||
      planParamsNorm.adjustmentsToSpending.extraSpending.discretionary.length >
        0

    const editRef = useRef<LabelAmountOptMonthRangeInputStateful>(null)
    return (
      <PlanInputBody
        {...props}
        onBackgroundClick={() => editRef.current?.closeSections()}
      >
        <div className="">
          <div
            className=""
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, { scale: 0.5 }),
            }}
          >
            <Contentful.RichText
              body={
                content['extra-spending'].intro[
                  planParamsNorm.advanced.strategy
                ]
              }
              p="p-base"
            />
            {showDiscretionary &&
              planParamsNorm.advanced.strategy === 'SWR' && (
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
                  planParamsNorm.advanced.strategy
                ]
              }
              p="p-base"
            />
            <LabeledAmountTimedListInput
              className="mt-6"
              editProps={{
                defaultAmountAndTiming,
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
                    planParamsNorm.advanced.strategy
                  ]
                }
                p="p-base"
              />
              <LabeledAmountTimedListInput
                className="mt-6"
                editProps={{
                  defaultAmountAndTiming,
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
              />
            </div>
          )}
        </div>
        {{
          input: editState
            ? (transitionOut) => (
                <LabelAmountOptMonthRangeInput
                  ref={editRef}
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
                  choicesPreFilter={{
                    start: ['now', 'retirement', 'numericAge', 'calendarMonth'],
                    end: [
                      'retirement',
                      'maxAge',
                      'numericAge',
                      'calendarMonth',
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
  ({
    forPrint = false,
    planParamsNorm,
  }: {
    forPrint?: boolean
    planParamsNorm: PlanParamsNormalized
  }) => {
    const { essential, discretionary } =
      planParamsNorm.adjustmentsToSpending.extraSpending
    const showLabels = planParamsNorm.advanced.strategy !== 'SWR'
    return (
      <>
        {essential.length === 0 && discretionary.length === 0 && <h2>None</h2>}
        {essential.length > 0 && (
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
            <PlanInputSummaryLabeledAmountTimedList entries={essential} />
          </>
        )}
        {discretionary.length > 0 && (
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
            <PlanInputSummaryLabeledAmountTimedList entries={discretionary} />
          </>
        )}
      </>
    )
  },
)
