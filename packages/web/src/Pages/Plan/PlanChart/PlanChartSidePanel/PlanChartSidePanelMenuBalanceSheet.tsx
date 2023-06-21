import { faAsterisk } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { ValueForMonthRange, fGet, noCase } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { getNetPresentValue } from '../../../../Utils/GetNetPresentValue'
import { useSimulation } from '../../../App/WithSimulation'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { Sankey } from './SankeyChart'

export const PlanChartSidePanelMenuBalanceSheet = React.memo(
  ({ show, onHide }: { show: boolean; onHide: () => void }) => {
    return (
      <CenteredModal
        className="w-[95vw] sm:w-[90vw] xl:w-[calc(1280px*0.9)]"
        show={show}
        onOutsideClickOrEscape={onHide}
      >
        <_Body />
      </CenteredModal>
    )
  },
)

const _Body = React.memo(() => {
  const { tpawResult } = useSimulation()
  const {
    sankeyModel,
    totalWealth,
    hasLegacy,
    estimatedCurrentPortfolioBalance,
    byMonthData,
    netPresentValue,
    generalSpending,
  } = useMemo(() => {
    const { params: paramsProcessed } = tpawResult
    const { netPresentValue, estimatedCurrentPortfolioBalance } =
      paramsProcessed
    const params = paramsProcessed.original

    const byMonthData = {
      wealth: {
        futureSavings: params.plan.wealth.futureSavings.map((x) =>
          _processValueForMonthRange(
            x,
            netPresentValue.tpaw.wealth.futureSavings,
          ),
        ),
        retirementIncome: params.plan.wealth.retirementIncome.map((x) =>
          _processValueForMonthRange(
            x,
            netPresentValue.tpaw.wealth.retirementIncome,
          ),
        ),
      },
      adjustmentsToSpending: {
        extraSpending: {
          essential:
            params.plan.adjustmentsToSpending.extraSpending.essential.map((x) =>
              _processValueForMonthRange(
                x,
                netPresentValue.tpaw.adjustmentsToSpending.extraSpending
                  .essential,
              ),
            ),
          discretionary:
            params.plan.adjustmentsToSpending.extraSpending.discretionary.map(
              (x) =>
                _processValueForMonthRange(
                  x,
                  netPresentValue.tpaw.adjustmentsToSpending.extraSpending
                    .discretionary,
                ),
            ),
        },
      },
    }
    const totalWealth = _.sum([
      estimatedCurrentPortfolioBalance,
      ...byMonthData.wealth.futureSavings.map((x) => x.netPresentValue),
      ...byMonthData.wealth.retirementIncome.map((x) => x.netPresentValue),
    ])
    const totalAdjustmentsToSpending = _.sum([
      ...byMonthData.adjustmentsToSpending.extraSpending.essential.map(
        (x) => x.netPresentValue,
      ),
      ...byMonthData.adjustmentsToSpending.extraSpending.discretionary.map(
        (x) => x.netPresentValue,
      ),
      netPresentValue.tpaw.adjustmentsToSpending.legacy,
    ])
    const generalSpending = totalWealth - totalAdjustmentsToSpending

    const c1 = {
      node: ChartUtils.color.amber[600],
      edge: ChartUtils.color.amber[400],
    }
    const c2 = {
      node: ChartUtils.color.purple[600],
      edge: ChartUtils.color.purple[400],
    }
    const c3 = {
      node: ChartUtils.color.red[600],
      edge: ChartUtils.color.red[400],
    }
    const colors0 = {
      futureSavingsParts: c1,
      retirementIncomeParts: c1,
      cpb: c1,
      futureSavingsTotal: c1,
      retirementIncomeTotal: c1,
      wealth: c1,
      generalSpending: c1,
      legacy: c1,
      essentialSpendingTotal: c1,
      discretionarySpendingTotal: c1,
      essentialSpendingParts: c1,
      discretionarySpendingParts: c1,
    }

    const colors1 = {
      futureSavingsParts: {
        node: ChartUtils.color.indigo[600],
        edge: ChartUtils.color.indigo[400],
      },
      retirementIncomeParts: {
        node: ChartUtils.color.indigo[600],
        edge: ChartUtils.color.indigo[400],
      },
      cpb: {
        node: ChartUtils.color.violet[500],
        edge: ChartUtils.color.violet[300],
      },
      futureSavingsTotal: {
        node: ChartUtils.color.violet[500],
        edge: ChartUtils.color.violet[300],
      },
      retirementIncomeTotal: {
        node: ChartUtils.color.violet[500],
        edge: ChartUtils.color.violet[300],
      },
      wealth: {
        node: ChartUtils.color.purple[800],
        edge: ChartUtils.color.purple[500],
      },
      generalSpending: {
        node: ChartUtils.color.pink[700],
        edge: ChartUtils.color.pink[500],
      },
      legacy: {
        node: ChartUtils.color.pink[700],
        edge: ChartUtils.color.pink[500],
      },
      essentialSpendingTotal: {
        node: ChartUtils.color.pink[700],
        edge: ChartUtils.color.pink[500],
      },
      discretionarySpendingTotal: {
        node: ChartUtils.color.pink[700],
        edge: ChartUtils.color.pink[500],
      },
      essentialSpendingParts: {
        node: ChartUtils.color.amber[700],
        edge: ChartUtils.color.amber[500],
      },
      discretionarySpendingParts: {
        node: ChartUtils.color.amber[700],
        edge: ChartUtils.color.amber[500],
      },
    }

    const colors = colors1

    const col1Nodes = {
      currentPortfolioBalance: {
        type: 'start',
        label: <h2 className="font-medium">Current Portfolio Balance</h2>,
        color: colors.cpb,
        hidden: true,
        quantity: estimatedCurrentPortfolioBalance,
      } as const,

      futureSavings: byMonthData.wealth.futureSavings
        .filter((x) => x.netPresentValue > 0)
        .map(
          (x): Sankey.NodeModel => ({
            type: 'start',
            label: <h2 className="font-medium">{x.label}</h2>,
            quantity: x.netPresentValue,
            hidden: false,
            color: colors.futureSavingsParts,
          }),
        ),
      retirementIncome: byMonthData.wealth.retirementIncome
        .filter((x) => x.netPresentValue > 0)
        .map(
          (x): Sankey.NodeModel => ({
            type: 'start',
            label: <h2 className="font-medium">{x.label}</h2>,
            color: colors.retirementIncomeParts,
            quantity: x.netPresentValue,
            hidden: false,
          }),
        ),
    }

    const noCol1 =
      col1Nodes.futureSavings.length === 0 &&
      col1Nodes.retirementIncome.length === 0

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
                  label: <h2 className="font-medium">Future Savings</h2>,
                  srcs: col1Nodes.futureSavings,
                  color: colors.futureSavingsTotal,
                  hidden: false,
                }
              : null,
          retirementIncome:
            col1Nodes.retirementIncome.length > 0
              ? {
                  type: 'merge' as const,
                  label: <h2 className="font-medium">Retirement Income</h2>,
                  srcs: col1Nodes.retirementIncome,
                  color: colors.retirementIncomeTotal,
                  hidden: false,
                }
              : null,
        }
    const col3Nodes: { wealth: Sankey.NodeModel } = {
      wealth: {
        type: 'merge' as const,
        label: (
          <div className="flex flex-col items-center justify-center">
            <h2 className="font-bol text-2xl whitespace-nowrap flex  items-center gap-x-4">
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
            <h2 className="font-medium text-[15px] mt-4">
              {formatCurrency(totalWealth)}
            </h2>
          </div>
        ),
        srcs: _.compact([
          col2Nodes.currentPortfolioBalance,
          col2Nodes.futureSavings,
          col2Nodes.retirementIncome,
        ]),
        color: colors.wealth,
        hidden: false,
      },
    }
    const hasCol5 =
      byMonthData.adjustmentsToSpending.extraSpending.essential.length > 0 ||
      byMonthData.adjustmentsToSpending.extraSpending.discretionary.length > 0
    const hasLegacy =
      tpawResult.params.adjustmentsToSpending.tpawAndSPAW.legacy.target > 0
    const col4Nodes = (() => {
      const [
        generalSpendingNode,
        legacyNode,
        essentialSpendingNode,
        discretionarySpendingNode,
      ] = Sankey.splitNode(col3Nodes.wealth, [
        {
          label: <h2 className="font-medium">General Spending</h2>,
          quantity: generalSpending,
          color: colors.generalSpending,
          hidden: false,
        },
        hasLegacy
          ? {
              label: <h2 className="font-medium">Legacy</h2>,
              quantity: netPresentValue.tpaw.adjustmentsToSpending.legacy,
              color: colors.legacy,
              hidden: false,
            }
          : null,
        byMonthData.adjustmentsToSpending.extraSpending.essential.length > 0
          ? {
              label: <h2 className="font-medium">Essential Spending</h2>,
              quantity: _.sumBy(
                byMonthData.adjustmentsToSpending.extraSpending.essential,
                (x) => x.netPresentValue,
              ),
              color: colors.essentialSpendingTotal,
              hidden: false,
            }
          : null,
        byMonthData.adjustmentsToSpending.extraSpending.discretionary.length > 0
          ? {
              label: <h2 className="font-medium">Discretionary Spending</h2>,
              quantity: _.sumBy(
                byMonthData.adjustmentsToSpending.extraSpending.discretionary,
                (x) => x.netPresentValue,
              ),
              color: colors.discretionarySpendingTotal,
              hidden: false,
            }
          : null,
      ])

      return {
        generalSpending: fGet(generalSpendingNode),
        legacy: legacyNode,
        essentialSpending: essentialSpendingNode,
        discretionarySpending: discretionarySpendingNode,
      }
    })()

    const col5Nodes = !hasCol5
      ? null
      : {
          generalSpending: { ...col4Nodes.generalSpending, hidden: true },
          legacy: col4Nodes.legacy
            ? { ...col4Nodes.legacy, hidden: true }
            : null,
          essentialSpending: col4Nodes.essentialSpending
            ? Sankey.splitNode(
                col4Nodes.essentialSpending,
                byMonthData.adjustmentsToSpending.extraSpending.essential.map(
                  (x) => ({
                    label: <h2 className="font-medium">{x.label}</h2>,
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
                    label: <h2 className="font-medium">{x.label}</h2>,
                    quantity: x.netPresentValue,
                    color: colors.discretionarySpendingParts,
                    hidden: false,
                  }),
                ),
              )
            : [],
        }
    const sankeyModel: Sankey.Model = _.compact([
      noCol1
        ? null
        : {
            labelPosition: 'left',
            nodes: [
              col1Nodes.currentPortfolioBalance,
              ...col1Nodes.futureSavings,
              ...col1Nodes.retirementIncome,
            ],
          },
      {
        labelPosition: 'left',
        nodes: _.compact([
          col2Nodes.currentPortfolioBalance,
          col2Nodes.futureSavings,
          col2Nodes.retirementIncome,
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
          col4Nodes.legacy,
          col4Nodes.essentialSpending,
          col4Nodes.discretionarySpending,
        ]),
      },
      col5Nodes
        ? {
            labelPosition: 'right',
            nodes: _.compact([
              col5Nodes.generalSpending,
              col5Nodes.legacy,
              ...col5Nodes.essentialSpending,
              ...col5Nodes.discretionarySpending,
            ]),
          }
        : null,
    ])
    return {
      sankeyModel,
      estimatedCurrentPortfolioBalance,
      byMonthData,
      netPresentValue,
      generalSpending,
      totalWealth,
      hasLegacy,
    }
  }, [tpawResult])
  return (
    <div className=" sm:px-4">
      <h2 className="font-bold text-3xl">Balance Sheet</h2>
      <div className="my-5 ">
        <Sankey.Chart
          className=""
          model={sankeyModel}
          heightOfMaxQuantity={150}
          vertAlign="top"
          sizing={{
            minNodeXGap: 190,
            maxNodeXGap: 500,
            nodeWidth: 10,
            nodeYGap: 30,
            paddingTop: 70,
            paddingBottom: 40,
          }}
        />
      </div>
      <div
        className="flex flex-col gap-y-16 md:grid gap-x-16"
        style={{ grid: 'auto/1fr 1fr' }}
      >
        <_Section
          className=""
          label="Wealth"
          total={totalWealth}
          asterix={false}
          parts={[
            {
              type: 'namedValue',
              label: 'Current Portfolio Balance',
              value: estimatedCurrentPortfolioBalance,
            },
            {
              type: 'valueForMonthRange',
              label: 'Future Savings',
              value: byMonthData.wealth.futureSavings,
            },
            {
              type: 'valueForMonthRange',
              label: 'Retirement Income',
              value: byMonthData.wealth.retirementIncome,
            },
          ]}
        />
        <_Section
          className=""
          label="Spending"
          total={totalWealth}
          asterix
          parts={[
            {
              type: 'namedValue',
              label: 'General Spending',
              value: generalSpending,
            },
            {
              type: 'namedValue',
              label: 'Legacy',
              value: hasLegacy
                ? netPresentValue.tpaw.adjustmentsToSpending.legacy
                : 'None',
            },
            {
              type: 'valueForMonthRange',
              label: 'Essential Spending',
              value: byMonthData.adjustmentsToSpending.extraSpending.essential,
            },
            {
              type: 'valueForMonthRange',
              label: 'Discretionary Spending',
              value:
                byMonthData.adjustmentsToSpending.extraSpending.discretionary,
            },
          ]}
        />
      </div>
      <div className="text-sm rounded-lg mt-10 ">
        <FontAwesomeIcon className="mb-1" icon={faAsterisk} />{' '}
        <span className="text-sm">{`The spending breakdown here reflects what spending would have been if it had not been constrained by things like ceilings, unfunded floors, and borrowing constraints which can move spending across different categories. For example, a ceiling can redirect spending from the general spending category to legacy, and that will not be reflected in this breakdown.`}</span>
      </div>
    </div>
  )
})

const _processValueForMonthRange = (
  x: ValueForMonthRange,
  netPresentValue: {
    byId: Record<number, ReturnType<typeof getNetPresentValue>>
  },
) => ({
  id: x.id,
  label: x.label ?? '<No label>',
  netPresentValue: netPresentValue.byId[x.id].withCurrentMonth[0],
})

const _Section = React.memo(
  ({
    label,
    className,
    total,
    parts,
    asterix,
  }: {
    label: string
    className?: string
    total: number
    parts: (
      | { type: 'namedValue'; label: string; value: number | 'None' }
      | {
          type: 'valueForMonthRange'
          label: string
          value: ReturnType<typeof _processValueForMonthRange>[]
        }
    )[]
    asterix: boolean
  }) => {
    return (
      <div className={clsx(className, 'border-gray-700 rounded-md')}>
        <h2 className="  text-2xl border-b-2 border-gray-700 pb-2 ">
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
              <h2 className="font-medium text-l mr-6">{x.label}</h2>
              {x.value === 'None' ? (
                <h2 className="text-sm lighten">None</h2>
              ) : (
                <h2 className="font-mono font-bold">
                  {formatCurrency(x.value)}
                </h2>
              )}
            </div>
          ) : x.type === 'valueForMonthRange' ? (
            <_ValueForMonthRangeDisplay
              key={i}
              className="mt-6"
              label={x.label}
              data={x.value}
            />
          ) : (
            noCase(x)
          ),
        )}

        <div className="flex justify-end items-center gap-x-4  pb-2 mt-6">
          <h2 className="text-lg"></h2>
          <h2 className="border-t-2  border-gray-700  pl-4 py-2  rounded-sm  font-mono">
            {formatCurrency(total)}
          </h2>
        </div>
      </div>
    )
  },
)

const _ValueForMonthRangeDisplay = React.memo(
  ({
    data,
    label,
    className,
  }: {
    data: ReturnType<typeof _processValueForMonthRange>[]
    label: string
    className?: string
  }) => {
    const total = _.sumBy(data, (x) => x.netPresentValue)
    return (
      <div className={clsx(className, ' rounded-lg ')}>
        <div className="flex justify-between ">
          <h2 className=" font-medium text-l mr-6">{label} </h2>
          {data.length > 0 ? (
            <h2 className=" font-mono font-bold">{formatCurrency(total)}</h2>
          ) : (
            <h2 className="lighten text-sm">None</h2>
          )}
        </div>
        {data.map((x, i) => (
          <_ValueForMonthRangeDisplayItem key={x.id} data={x} />
        ))}
      </div>
    )
  },
)
const _ValueForMonthRangeDisplayItem = React.memo(
  ({ data }: { data: ReturnType<typeof _processValueForMonthRange> }) => {
    return (
      <div className="flex justify-between pl-4 pt-2 lighten">
        <h2 className="mr-6">{data.label}</h2>
        <h2 className="font-mono text-sm ">
          {formatCurrency(data.netPresentValue)}
        </h2>
      </div>
    )
  },
)
