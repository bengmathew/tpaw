import { faAsterisk } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { LabeledAmountTimed, fGet, noCase } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { useMemo } from 'react'

import { formatCurrency } from '../../../../../../Utils/FormatCurrency'
import { getNetPresentValue } from '../../../../../../Utils/GetNetPresentValue'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../../UsePlanColors'
import { Sankey } from './SankeyChart'
import {
  indigo,
  violet,
  fuchsia,
  red,
  amber,
} from '../../../../../../Utils/ColorPalette'
import clsx from 'clsx'
import { NormalizedLabeledAmountTimed } from '@tpaw/common'

export const PlanResultsSidePanelMenuBalanceSheet = React.memo(
  ({ show, onHide }: { show: boolean; onHide: () => void }) => {
    const planColors = usePlanColors()
    return (
      <CenteredModal
        className="w-[calc(100vw-20px)] max-w-[1200px] "
        show={show}
        onOutsideClickOrEscape={onHide}
        style={{
          backgroundColor: planColors.results.cardBG,
          color: planColors.results.fg,
        }}
      >
        <BalanceSheetContent className="sm:px-4" />
      </CenteredModal>
    )
  },
)

export const BalanceSheetContent = React.memo(
  ({
    className,
    forPrint = false,
  }: {
    className?: string
    forPrint?: boolean
  }) => {
    const { simulationResult } = useSimulationResultInfo()
    const {
      portfolioBalance,
      sankeyModel,
      totalWealth,
      hasLegacy,
      byMonthData,
      netPresentValue,
      generalSpending,
    } = useMemo(() => {
      const {
        planParamsProcessed,
        planParamsNormOfResult,
        portfolioBalanceEstimationByDated,
        tpawApproxNetPresentValueForBalanceSheet,
      } = simulationResult
      const netPresentValue = fGet(tpawApproxNetPresentValueForBalanceSheet)
      const portfolioBalance = portfolioBalanceEstimationByDated.currentBalance

      const _filter = (x: NormalizedLabeledAmountTimed) =>
        x.amountAndTiming.type !== 'inThePast'
      const byMonthData = {
        wealth: {
          futureSavings: planParamsNormOfResult.wealth.futureSavings
            .filter(_filter)
            .map((x) =>
              _processLabeledAmountTimed(x, netPresentValue.futureSavings),
            ),
          incomeDuringRetirement:
            planParamsNormOfResult.wealth.incomeDuringRetirement
              .filter(_filter)
              .map((x) =>
                _processLabeledAmountTimed(
                  x,
                  netPresentValue.incomeDuringRetirement,
              ),
            ),
        },
        adjustmentsToSpending: {
          extraSpending: {
            essential:
              planParamsNormOfResult.adjustmentsToSpending.extraSpending.essential
                .filter(_filter)
                .map((x) =>
                  _processLabeledAmountTimed(
                    x,
                    netPresentValue.essentialExpenses,
                  ),
              ),
            discretionary:
              planParamsNormOfResult.adjustmentsToSpending.extraSpending.discretionary
                .filter(_filter)
                .map((x) =>
                  _processLabeledAmountTimed(
                    x,
                    netPresentValue.discretionaryExpenses,
                  ),
              ),
          },
        },
      }
      const totalWealth = _.sum([
        portfolioBalanceEstimationByDated.currentBalance,
        ...byMonthData.wealth.futureSavings.map((x) => x.netPresentValue),
        ...byMonthData.wealth.incomeDuringRetirement.map(
          (x) => x.netPresentValue,
        ),
      ])
      const totalAdjustmentsToSpending = _.sum([
        ...byMonthData.adjustmentsToSpending.extraSpending.essential.map(
          (x) => x.netPresentValue,
        ),
        ...byMonthData.adjustmentsToSpending.extraSpending.discretionary.map(
          (x) => x.netPresentValue,
        ),
        netPresentValue.legacyTarget,
      ])
      const generalSpending = totalWealth - totalAdjustmentsToSpending

      const col1Color = {
        node: indigo[500],
        edge: indigo[300],
      }
      const col2Color = {
        node: violet[500],
        edge: violet[300],
      }
      const col3Color = {
        node: fuchsia[600],
        edge: fuchsia[400],
      }
      const col4Color = {
        node: red[600],
        edge: red[400],
      }
      const col5Color = {
        node: amber[600],
        edge: amber[400],
      }
      const colors = {
        futureSavingsParts: col1Color,
        incomeDuringRetirementParts: col1Color,
        cpb: col2Color,
        futureSavingsTotal: col2Color,
        incomeDuringRetirementTotal: col2Color,
        wealth: col3Color,
        generalSpending: col4Color,
        legacy: col4Color,
        essentialSpendingTotal: col4Color,
        discretionarySpendingTotal: col4Color,
        essentialSpendingParts: col5Color,
        discretionarySpendingParts: col5Color,
      }

      const col1Nodes = {
        currentPortfolioBalance: {
          type: 'start',
          label: (
            <_ChartLabel
              label="Current Portfolio Balance"
              forPrint={forPrint}
            />
          ),
          color: colors.cpb,
          hidden: true,
          quantity: portfolioBalanceEstimationByDated.currentBalance,
        } as const,

        futureSavings: byMonthData.wealth.futureSavings
          .filter((x) => x.netPresentValue > 0)
          .map(
            (x): Sankey.NodeModel => ({
              type: 'start',
              label: <_ChartLabel label={x.label} forPrint={forPrint} />,
              quantity: x.netPresentValue,
              hidden: false,
              color: colors.futureSavingsParts,
            }),
          ),
        incomeDuringRetirement: byMonthData.wealth.incomeDuringRetirement
          .filter((x) => x.netPresentValue > 0)
          .map(
            (x): Sankey.NodeModel => ({
              type: 'start',
              label: <_ChartLabel label={x.label} forPrint={forPrint} />,
              color: colors.incomeDuringRetirementParts,
              quantity: x.netPresentValue,
              hidden: false,
            }),
          ),
      }

      const noCol1 =
        col1Nodes.futureSavings.length === 0 &&
        col1Nodes.incomeDuringRetirement.length === 0

      const col2Nodes = noCol1
        ? {
            currentPortfolioBalance: {
              ...col1Nodes.currentPortfolioBalance,
              hidden: false,
            },
          }
        : {
            currentPortfolioBalance: {
              ...col1Nodes.currentPortfolioBalance,
              hidden: false,
            },
            futureSavings:
              col1Nodes.futureSavings.length > 0
                ? {
                    type: 'merge' as const,
                    label: (
                      <_ChartLabel label="Future Savings" forPrint={forPrint} />
                    ),
                    srcs: col1Nodes.futureSavings,
                    color: colors.futureSavingsTotal,
                    hidden: false,
                  }
                : null,
            incomeDuringRetirement:
              col1Nodes.incomeDuringRetirement.length > 0
                ? {
                    type: 'merge' as const,
                    label: (
                      <_ChartLabel
                        label="Income During Retirement"
                        forPrint={forPrint}
                      />
                    ),
                    srcs: col1Nodes.incomeDuringRetirement,
                    color: colors.incomeDuringRetirementTotal,
                    hidden: false,
                  }
                : null,
          }
      const col3Nodes: { wealth: Sankey.NodeModel } = {
        wealth: {
          type: 'merge' as const,
          label: (
            <div className="flex flex-col items-center justify-center">
              <h2 className="font-bold text-xl sm:text-2xl whitespace-nowrap flex  items-center gap-x-4">
                <span className="w-[100px] text-right">Wealth</span>
                <div className="block border-r border-gray-400 h-[30px] " />
                <span className="w-[100px]">
                  Spending
                  <FontAwesomeIcon
                    className="text-sm mb-2 ml-1"
                    icon={faAsterisk}
                  />
                </span>
              </h2>
              {forPrint && <h2 className="text-[15px]">Net Present Value</h2>}

              <h2
                className={clsx(
                  'font-medium text-[15px]',
                  forPrint ? 'mt-2' : 'mt-4',
                )}
              >
                {formatCurrency(totalWealth)}
              </h2>
            </div>
          ),
          srcs: _.compact([
            col2Nodes.currentPortfolioBalance,
            col2Nodes.futureSavings,
            col2Nodes.incomeDuringRetirement,
          ]),
          color: colors.wealth,
          hidden: false,
        },
      }
      const hasCol5 =
        byMonthData.adjustmentsToSpending.extraSpending.essential.length > 0 ||
        byMonthData.adjustmentsToSpending.extraSpending.discretionary.length > 0
      const hasLegacy = netPresentValue.legacyTarget > 0
      const col4Nodes = (() => {
        const [
          generalSpendingNode,
          essentialSpendingNode,
          discretionarySpendingNode,
          legacyNode,
        ] = Sankey.splitNode(col3Nodes.wealth, [
          {
            label: <_ChartLabel label="General Spending" forPrint={forPrint} />,
            quantity: generalSpending,
            color: colors.generalSpending,
            hidden: false,
          },

          byMonthData.adjustmentsToSpending.extraSpending.essential.length > 0
            ? {
                label: (
                  <_ChartLabel label="Essential Spending" forPrint={forPrint} />
                ),
                quantity: _.sumBy(
                  byMonthData.adjustmentsToSpending.extraSpending.essential,
                  (x) => x.netPresentValue,
                ),
                color: colors.essentialSpendingTotal,
                hidden: false,
              }
            : null,
          byMonthData.adjustmentsToSpending.extraSpending.discretionary.length >
          0
            ? {
                label: (
                  <_ChartLabel
                    label="Discretionary Spending"
                    forPrint={forPrint}
                  />
                ),
                quantity: _.sumBy(
                  byMonthData.adjustmentsToSpending.extraSpending.discretionary,
                  (x) => x.netPresentValue,
                ),
                color: colors.discretionarySpendingTotal,
                hidden: false,
              }
            : null,
          hasLegacy
            ? {
                label: <_ChartLabel label="Legacy" forPrint={forPrint} />,
                quantity: netPresentValue.legacyTarget,
                color: colors.legacy,
                hidden: false,
              }
            : null,
        ])

        return {
          generalSpending: fGet(generalSpendingNode),
          essentialSpending: essentialSpendingNode,
          discretionarySpending: discretionarySpendingNode,
          legacy: legacyNode,
        }
      })()

      const col5Nodes = !hasCol5
        ? null
        : {
            generalSpending: { ...col4Nodes.generalSpending, hidden: true },

            essentialSpending: col4Nodes.essentialSpending
              ? Sankey.splitNode(
                  col4Nodes.essentialSpending,
                  byMonthData.adjustmentsToSpending.extraSpending.essential.map(
                    (x) => ({
                      label: (
                        <_ChartLabel label={x.label} forPrint={forPrint} />
                      ),
                      quantity: x.netPresentValue,
                      color: colors.essentialSpendingParts,
                      hidden: false,
                    }),
                  ),
                )
              : [],
            discretionarySpending: col4Nodes.discretionarySpending
              ? Sankey.splitNode(
                  col4Nodes.discretionarySpending,
                  byMonthData.adjustmentsToSpending.extraSpending.discretionary.map(
                    (x) => ({
                      label: (
                        <_ChartLabel label={x.label} forPrint={forPrint} />
                      ),
                      quantity: x.netPresentValue,
                      color: colors.discretionarySpendingParts,
                      hidden: false,
                    }),
                  ),
                )
              : [],
            legacy: col4Nodes.legacy
              ? { ...col4Nodes.legacy, hidden: true }
              : null,
          }
      const sankeyModel: Sankey.Model = _.compact([
        {
          labelPosition: 'left',
          nodes: noCol1
            ? []
            : [
                col1Nodes.currentPortfolioBalance,
                ...col1Nodes.futureSavings,
                ...col1Nodes.incomeDuringRetirement,
              ],
        },
        {
          labelPosition: 'left',
          nodes: _.compact([
            col2Nodes.currentPortfolioBalance,
            col2Nodes.futureSavings,
            col2Nodes.incomeDuringRetirement,
          ]),
        },
        {
          labelPosition: 'top',
          nodes: [col3Nodes.wealth],
        },
        {
          labelPosition: 'right',
          nodes: _.compact([
            col4Nodes.generalSpending,
            col4Nodes.essentialSpending,
            col4Nodes.discretionarySpending,
            col4Nodes.legacy,
          ]),
        },
        {
          labelPosition: 'right',
          nodes: col5Nodes
            ? _.compact([
                col5Nodes.generalSpending,
                ...col5Nodes.essentialSpending,
                ...col5Nodes.discretionarySpending,
                col5Nodes.legacy,
              ])
            : [],
        },
      ])
      // If start and end are empty, remove both. We display the empty columns
      // only for centering, so removing them symmetrically keeps the centering
      // while not having dead space on either side if not needed.
      while (
        fGet(_.first(sankeyModel)).nodes.length === 0 &&
        fGet(_.last(sankeyModel)).nodes.length === 0
      ) {
        sankeyModel.pop()
        sankeyModel.shift()
      }

      return {
        sankeyModel,
        byMonthData,
        netPresentValue,
        generalSpending,
        totalWealth,
        portfolioBalance,
        hasLegacy,
      }
    }, [forPrint, simulationResult])
    return (
      <div className={clix(className)}>
        {!forPrint && (
          <>
            <h2 className="font-bold text-2xl sm:text-3xl">Balance Sheet</h2>
            <h2 className="mb-5">Net Present Value</h2>
          </>
        )}
        <div className=" ">
          <Sankey.Chart
            className=""
            model={sankeyModel}
            heightOfMaxQuantity={150}
            vertAlign="top"
            sizing={{
              minNodeXGap: forPrint ? 0 : 190,
              nodeWidth: 10,
              nodeYGap: 30,
              paddingTop: forPrint ? 85 : 70,
              paddingBottom: 40,
            }}
          />
        </div>
        <div
          className={clix(
            forPrint ? 'grid' : 'flex flex-col gap-y-16 md:grid',
            'gap-x-16',
          )}
          style={{ grid: 'auto/1fr 1fr' }}
        >
          <_Section
            className=""
            label="Wealth"
            total={totalWealth}
            asterix={false}
            forPrint={forPrint}
            parts={[
              {
                type: 'namedValue',
                label: 'Current Portfolio Balance',
                value: portfolioBalance,
              },
              {
                type: 'valueForMonthRange',
                label: 'Future Savings',
                value: byMonthData.wealth.futureSavings,
              },
              {
                type: 'valueForMonthRange',
                label: 'Income During Retirement',
                value: byMonthData.wealth.incomeDuringRetirement,
              },
            ]}
          />
          <_Section
            className=""
            label="Spending"
            total={totalWealth}
            asterix
            forPrint={forPrint}
            parts={[
              {
                type: 'namedValue',
                label: 'General Spending',
                value: generalSpending,
              },

              {
                type: 'valueForMonthRange',
                label: 'Essential Spending',
                value:
                  byMonthData.adjustmentsToSpending.extraSpending.essential,
              },
              {
                type: 'valueForMonthRange',
                label: 'Discretionary Spending',
                value:
                  byMonthData.adjustmentsToSpending.extraSpending.discretionary,
              },
              {
                type: 'namedValue',
                label: 'Legacy',
                value: hasLegacy ? netPresentValue.legacyTarget : 'None',
              },
            ]}
          />
        </div>
        <div className={clix('rounded-lg mt-10 ')}>
          <FontAwesomeIcon
            className={clix(forPrint ? 'text-[10px]' : 'text-[11px]', ' mb-1')}
            icon={faAsterisk}
          />{' '}
          <span className="">{`This spending breakdown reflects what spending would have been had it not been constrained by things like ceilings, unfunded floors, and borrowing constraints which can move spending across different categories. For example, a ceiling can redirect spending from the general spending category to legacy, and that will not be reflected in this breakdown.`}</span>
        </div>
      </div>
    )
  },
)

const _processLabeledAmountTimed = (
  x: NormalizedLabeledAmountTimed,
  netPresentValue: { id: string; value: number }[],
) => ({
  id: x.id,
  label: x.label ?? '<No label>',
  netPresentValue: fGet(netPresentValue.find(({ id }) => id === x.id)).value,
})

const _Section = React.memo(
  ({
    label,
    className,
    total,
    parts,
    asterix,
    forPrint,
  }: {
    label: string
    className?: string
    total: number
    parts: (
      | { type: 'namedValue'; label: string; value: number | 'None' }
      | {
          type: 'valueForMonthRange'
          label: string
          value: ReturnType<typeof _processLabeledAmountTimed>[]
        }
    )[]
    asterix: boolean
    forPrint: boolean
  }) => {
    return (
      <div className={clix(className, 'border-gray-700 rounded-md')}>
        <h2 className=" font-bold text-xl sm:text-2xl border-b-2 border-gray-700 pb-2 ">
          {label}
          {asterix && (
            <FontAwesomeIcon className="text-sm mb-2 ml-1" icon={faAsterisk} />
          )}
        </h2>
        {parts.map((x, i) =>
          x.type === 'namedValue' ? (
            <div
              key={i}
              className="rounded-lg flex  justify-between items-center mt-6"
            >
              <h2 className="font-medium mr-6">{x.label}</h2>
              {x.value === 'None' ? (
                <h2
                  className={clix(
                    forPrint ? 'text-[11px]' : 'text-sm',
                    'lighten',
                  )}
                >
                  None
                </h2>
              ) : (
                <h2 className="font-mono font-bold">
                  {formatCurrency(x.value)}
                </h2>
              )}
            </div>
          ) : x.type === 'valueForMonthRange' ? (
            <_LabeledAmountTimedDisplay
              key={i}
              className="mt-6"
              label={x.label}
              data={x.value}
              forPrint={forPrint}
            />
          ) : (
            noCase(x)
          ),
        )}

        <div className="flex justify-end items-center gap-x-4  pb-2 mt-6">
          <h2 className="text-lg"></h2>
          <h2 className="border-t-2  border-gray-700  pl-4 py-2  rounded-xs  font-mono">
            {formatCurrency(total)}
          </h2>
        </div>
      </div>
    )
  },
)

const _LabeledAmountTimedDisplay = React.memo(
  ({
    data,
    label,
    className,
    forPrint,
  }: {
    data: ReturnType<typeof _processLabeledAmountTimed>[]
    label: string
    className?: string
    forPrint: boolean
  }) => {
    const total = _.sumBy(data, (x) => x.netPresentValue)
    return (
      <div className={clix(className, ' rounded-lg ')}>
        <div className="flex justify-between ">
          <h2 className=" font-medium text-l mr-6">{label} </h2>
          {data.length > 0 ? (
            <h2 className=" font-mono font-bold">{formatCurrency(total)}</h2>
          ) : (
            <h2
              className={clix(forPrint ? 'text-[11px]' : 'text-sm', 'lighten')}
            >
              None
            </h2>
          )}
        </div>
        {data.map((x, i) => (
          <_LabeledAmountTimedDisplayItem
            key={x.id}
            data={x}
            forPrint={forPrint}
          />
        ))}
      </div>
    )
  },
)
const _LabeledAmountTimedDisplayItem = React.memo(
  ({
    data,
    forPrint,
  }: {
    data: ReturnType<typeof _processLabeledAmountTimed>
    forPrint: boolean
  }) => {
    return (
      <div className="flex justify-between pl-4 pt-2 lighten">
        <h2 className="mr-6">{data.label}</h2>
        <h2 className={clix(forPrint ? 'text-[10px]' : 'text-sm', 'font-mono')}>
          {formatCurrency(data.netPresentValue)}
        </h2>
      </div>
    )
  },
)

const _ChartLabel = React.memo(
  ({
    className,
    label,
    forPrint,
  }: {
    className?: string
    label: string
    forPrint: boolean
  }) => {
    return (
      <h2
        className={clix(
          className,
          // Thanks: https://stackoverflow.com/a/7993098
          ' whitespace-nowrap text-ellipsis overflow-hidden w-full',
          forPrint ? 'text-[10.5px]' : 'text-[12px] font-medium',
        )}
      >
        {label}
      </h2>
    )
  },
)
