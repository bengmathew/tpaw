import { faCaretRight, faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { fGet, noCase } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import Link from 'next/link'
import React, { CSSProperties, useEffect, useRef, useState } from 'react'
import { Padding, RectExt, rectExt } from '../../../../../Utils/Geometry'
import { useSystemInfo } from '../../../../App/WithSystemInfo'
import {
  ChartReact,
  ChartReactStatefull,
} from '../../../../Common/Chart/ChartReact'
import { ContextModal } from '../../../../Common/Modal/ContextModal'
import { Config } from '../../../../Config'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { useChartData } from '../../WithPlanResultsChartData'
import { PlanResultsTransitionState } from '../PlanResults'
import { PlanResultsChartType } from '../PlanResultsChartType'
import { useGetPlanResultsChartURL } from '../UseGetPlanResultsChartURL'
import { getPlanResultsChartRange } from './PlanResultsChart/GetPlanResultsChartRange'
import { PlanResultsChartData } from './PlanResultsChart/PlanResultsChartData'
import { PlanResultsChartCardMenuButton } from './PlanResultsChartCardMenuButton'
import {
  getPlanResultsChartLabelInfoForSpending,
  planResultsChartLabel,
} from './PlanResultsChartLabel'
import { NormalizedLabeledAmountTimed } from '@tpaw/common'

const maxWidth = 700
export type PlanChartMainCardMenuStateful = {
  setButtonScale: (scale: number) => void
}
export const PlanResultsChartCardMenu = React.memo(
  ({
    className = '',
    style,
    transition,
  }: {
    className?: string
    style?: CSSProperties
    transition: { target: PlanResultsTransitionState; duration: number }
  }) => {
    const { windowWidthName } = useSystemInfo()
    const { simulationResult } = useSimulationResultInfo()
    const { planParamsNormOfResult } = simulationResult
    const planColors = usePlanColors()
    const { nonPlanParams } = useNonPlanParams()

    const [expandEssential, setExpandEssential] = useState(false)
    const [expandDiscretionary, setExpandDiscretionary] = useState(false)

    const chartH = windowWidthName === 'xs' ? 50 : 55

    const isNotInThePast = ({
      amountAndTiming,
    }: NormalizedLabeledAmountTimed) => {
      switch (amountAndTiming.type) {
        case 'inThePast':
          return false
        case 'oneTime':
        case 'recurring':
          return true
        default:
          noCase(amountAndTiming)
      }
    }

    const essentialArray =
      planParamsNormOfResult.adjustmentsToSpending.extraSpending.essential
        .filter(isNotInThePast)
        .sort((a, b) => a.sortIndex - b.sortIndex)

    const discretionaryArray =
      planParamsNormOfResult.adjustmentsToSpending.extraSpending.discretionary
        .filter(isNotInThePast)
        .sort((a, b) => a.sortIndex - b.sortIndex)

    const spendingLabelInfo = getPlanResultsChartLabelInfoForSpending(
      planParamsNormOfResult,
    )
    const marginToWindow = windowWidthName === 'xs' ? 0 : 10
    return (
      <Menu>
        {({ open }) => (
          <ContextModal
            align={'left'}
            open={open}
            afterLeave={() => {
              setExpandEssential(false)
              setExpandDiscretionary(false)
            }}
            getMarginToWindow={() => marginToWindow}
            onOutsideClickOrEscape={null}
          >
            {({ ref }) => (
              <Menu.Button ref={ref}>
                <PlanResultsChartCardMenuButton
                  ref={ref}
                  className={className}
                  style={style}
                  transition={transition}
                />
              </Menu.Button>
            )}

            <Menu.Items
              className="rounded-lg"
              style={{
                maxWidth: `calc(100vw - ${marginToWindow * 2}px)`,
                width: '700px',
                color: planColors.results.fg,
              }}
            >
              <_Link type="spending-total" indent={0} chartH={chartH} />
              {spendingLabelInfo.hasExtra && (
                <>
                  <_Link indent={1} type="spending-general" chartH={chartH} />
                  {spendingLabelInfo.extraSpendingLabelInfo
                    .splitEssentialAndDiscretionary ? (
                    <>
                      {essentialArray.length > 0 && (
                        <>
                          <_Button
                            indent={1}
                            isExpanded={expandEssential}
                            onClick={() => setExpandEssential((x) => !x)}
                            label={
                              spendingLabelInfo.extraSpendingLabelInfo.essential
                                .label
                            }
                            description={
                              spendingLabelInfo.extraSpendingLabelInfo.essential
                                .description
                            }
                            chartH={chartH}
                          />
                          {expandEssential &&
                            essentialArray.map((x) => (
                              <_Link
                                indent={2}
                                key={`essential-${x.id}`}
                                type={`spending-essential-${x.id}`}
                                chartH={chartH}
                              />
                            ))}
                        </>
                      )}
                      {discretionaryArray.length > 0 && (
                        <>
                          <_Button
                            isExpanded={expandDiscretionary}
                            indent={1}
                            onClick={() => setExpandDiscretionary((x) => !x)}
                            label={
                              spendingLabelInfo.extraSpendingLabelInfo
                                .discretionary.label
                            }
                            description={
                              spendingLabelInfo.extraSpendingLabelInfo
                                .discretionary.description
                            }
                            chartH={chartH}
                          />
                          {expandDiscretionary &&
                            discretionaryArray.map((x) => (
                              <_Link
                                indent={2}
                                key={`discretionary-${x.id}`}
                                type={`spending-discretionary-${x.id}`}
                                chartH={chartH}
                              />
                            ))}
                        </>
                      )}
                    </>
                  ) : (
                    <>
                      <_Button
                        isExpanded={expandEssential}
                        onClick={() => setExpandEssential((x) => !x)}
                        label={spendingLabelInfo.extraSpendingLabelInfo.label}
                        description={
                          spendingLabelInfo.extraSpendingLabelInfo.description
                        }
                        indent={1}
                        chartH={chartH}
                      />
                      {expandEssential &&
                        [
                          ...planParamsNormOfResult.adjustmentsToSpending
                            .extraSpending.essential,
                        ]
                          .sort((a, b) => a.sortIndex - b.sortIndex)
                          .map((x) => (
                            <_Link
                              key={`essential-${x.id}`}
                              indent={2}
                              type={`spending-essential-${x.id}`}
                              chartH={chartH}
                            />
                          ))}
                      {expandEssential &&
                        [
                          ...planParamsNormOfResult.adjustmentsToSpending
                            .extraSpending.discretionary,
                        ]
                          .sort((a, b) => a.sortIndex - b.sortIndex)
                          .map((x) => (
                            <_Link
                              key={`discretionary-${x.id}`}
                              indent={2}
                              type={`spending-discretionary-${x.id}`}
                              chartH={chartH}
                            />
                          ))}
                    </>
                  )}
                </>
              )}

              <_Link type={'portfolio'} indent={0} chartH={chartH} />
              <_Link
                type={'asset-allocation-savings-portfolio'}
                indent={0}
                chartH={chartH}
              />
              <_Link type={'withdrawal'} indent={0} chartH={chartH} />
              {(!Config.client.isProduction ||
                nonPlanParams.dev.showDevFeatures) && (
                <_Link
                  type="asset-allocation-total-portfolio"
                  indent={0}
                  chartH={chartH}
                />
              )}
            </Menu.Items>
          </ContextModal>
        )}
      </Menu>
    )
  },
)

const _DescriptionLine = React.memo(
  ({ description }: { description: string }) => (
    <p className={clix(` text-sm -mt-1`, 'lighten-2')}>{description}</p>
  ),
)

const _LabelLine = React.memo(({ label }: { label: readonly string[] }) => (
  <h2 className={clix('text-base sm:text-lg font-bold')}>
    {label.map((x, i) => (
      <React.Fragment key={i}>
        <span>{x}</span>
        {i !== label.length - 1 && (
          <FontAwesomeIcon
            className="mx-2 text-xs sm:text-sm lighten-2"
            icon={faChevronRight}
          />
        )}
      </React.Fragment>
    ))}
  </h2>
))

const _Button = React.memo(
  ({
    onClick,
    isExpanded,
    label,
    description,
    chartH,
    indent,
  }: {
    isExpanded: boolean
    onClick: () => void
    label: readonly string[]
    description: string
    chartH: number
    indent: 0 | 1 | 2
  }) => {
    const planColors = usePlanColors()
    return (
      <Menu.Item>
        {({ active }) => (
          <div className="px-2 py-1">
            <button
              className={clix(
                indent === 0
                  ? 'pl-2'
                  : indent === 1
                    ? 'pl-8'
                    : indent === 2
                      ? 'pl-16'
                      : noCase(indent),
                'text-start w-full  rounded-lg pr-2 py-1',
              )}
              style={{
                backgroundColor: active
                  ? planColors.shades.alt[5].hex
                  : 'rgb(0,0,0,0)',
              }}
              onClick={(e) => {
                // This keeps the menu open (only on click though, not on keyboard)
                // As of Jun 2023, no solution for keyboard:
                // https://github.com/tailwindlabs/headlessui/discussions/1122
                e.preventDefault()
                onClick()
              }}
            >
              <div
                className="flex items-center"
                style={{ minHeight: `${chartH}px` }}
              >
                <div className="">
                  <div className="flex items-center gap-x-2">
                    <_LabelLine label={label} />
                    <FontAwesomeIcon
                      className={clix(
                        'text-xl transition-transform duration-300',
                        isExpanded && 'rotate-90',
                      )}
                      icon={faCaretRight}
                    />
                  </div>
                  <_DescriptionLine description={description} />
                </div>
              </div>
            </button>
          </div>
        )}
      </Menu.Item>
    )
  },
)

const _Link = React.memo(
  ({
    type,
    chartH,
    indent,
  }: {
    type: PlanResultsChartType
    indent: 0 | 1 | 2
    chartH: number
  }) => {
    const { simulationResult } = useSimulationResultInfo()
    const { planParamsNormOfResult } = simulationResult
    const getPlanChartURL = useGetPlanResultsChartURL()
    const chartData = useChartData(type)

    const { label, description } = planResultsChartLabel(
      planParamsNormOfResult,
      type,
    )
    const { windowWidthName } = useSystemInfo()
    const width = windowWidthName === 'xs' ? 120 : 145

    const planColors = usePlanColors()
    return (
      <Menu.Item>
        {({ active }) => (
          <Link
            className={clix('block px-2 py-1')}
            href={getPlanChartURL(type)}
            shallow
          >
            <div
              className={clix(
                indent === 0
                  ? 'pl-2'
                  : indent === 1
                    ? 'pl-8'
                    : indent === 2
                      ? 'pl-16'
                      : noCase(indent),
                'grid gap-x-4 items-center  rounded-lg pr-2 py-1',
              )}
              style={{
                transitionProperty: 'background-color',
                transitionDuration: '300ms',
                grid: 'auto / 1fr auto',
                backgroundColor: active
                  ? planColors.shades.alt[5].hex
                  : 'rgb(0,0,0,0)',
              }}
            >
              <div className="">
                <_LabelLine label={label.forMenu} />
                <_DescriptionLine description={description} />
              </div>
              <div
                className={`relative  rounded-xl  flex flex-col justify-center `}
                style={{
                  width: `${width}px`,
                  height: `${chartH}px`,
                  backgroundColor: planColors.shades.alt[2].hex,
                }}
              >
                <_Chart
                  data={chartData}
                  startingSizing={{
                    position: rectExt({ x: 0, y: 0, width, height: chartH }),
                    padding: { left: 10, right: 10, top: 5, bottom: 5 },
                  }}
                />
              </div>
            </div>
          </Link>
        )}
      </Menu.Item>
    )
  },
)

const _Chart = React.memo(
  ({
    data,
    startingSizing,
  }: {
    data: PlanResultsChartData
    startingSizing: { position: RectExt; padding: Padding }
  }) => {
    const ref =
      useRef<ChartReactStatefull<{ data: PlanResultsChartData }>>(null)
    useEffect(() => {
      if (!ref.current) return
      ref.current.setData({ data }, null)
    }, [data])

    return (
      <ChartReact<{ data: PlanResultsChartData }>
        ref={ref}
        starting={{
          data: { data },
          sizing: startingSizing,
          propsFn: ({ data }) => ({
            dataRange: {
              x: data.displayRange.x,
              y: data.displayRange.y,
            },
            includeWidthOfLastX: false,
          }),
        }}
        components={components}
      />
    )
  },
)

const components = () => [getPlanResultsChartRange('menu')]
