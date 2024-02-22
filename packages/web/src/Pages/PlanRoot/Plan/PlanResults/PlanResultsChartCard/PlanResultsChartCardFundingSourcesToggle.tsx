import { block } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { CSSProperties } from 'react'
import { PERCENTILES_STR } from '../../../../../UseSimulator/Simulator/Simulator'
import { useURLUpdater } from '../../../../../Utils/UseURLUpdater'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
import { usePlanColors } from '../../UsePlanColors'
import {
    getPlanResultsChartSpendingTotalFundingSourcesPercentile,
    isPlanResultsChartSpendingTotalFundingSourcesType,
} from '../PlanResultsChartType'
import { useGetPlanResultsChartURL } from '../UseGetPlanResultsChartURL'
import { usePlanResultsChartType } from '../UsePlanResultsChartType'

export const PlanResultsChartCardFundingSourcesToggle = React.memo(
  ({
    className,
    style,
    layout,
  }: {
    className?: string
    style?: CSSProperties
    layout: 'laptop' | 'desktop' | 'mobile'
  }) => {
    const planColors = usePlanColors()
    const getPlanChartURL = useGetPlanResultsChartURL()
    const chartType = usePlanResultsChartType()
    const urlUpdater = useURLUpdater()

    const currPrecentile = isPlanResultsChartSpendingTotalFundingSourcesType(
      chartType,
    )
      ? getPlanResultsChartSpendingTotalFundingSourcesPercentile(chartType)
      : '50'
    const isShowingFundingSources =
      isPlanResultsChartSpendingTotalFundingSourcesType(chartType)

    const handleDelta = (delta: 1 | -1) => {
      urlUpdater.push(
        getPlanChartURL(
          `spending-total-funding-sources-${
            PERCENTILES_STR[
              _.clamp(
                PERCENTILES_STR.indexOf(currPrecentile) + delta,
                0,
                PERCENTILES_STR.length - 1,
              )
            ]
          }`,
        ),
      )
    }

    return (
      <div className={clix(className, 'flex justify-start')} style={style}>
        <div className="flex items-center gap-x-2">
          <ToggleSwitch
            style={{
              bg: {
                on: { color: planColors.shades.main[13].hex },
                off: { color: planColors.shades.light[5].hex },
              },
            }}
            sizing={
              layout === 'mobile'
                ? { width: 30, height: 17, gap: 2.5 }
                : { width: 34, height: 18, gap: 2.5 }
            }
            checked={isShowingFundingSources}
            setChecked={(checked) =>
              urlUpdater.push(
                getPlanChartURL(
                  !checked
                    ? 'spending-total'
                    : `spending-total-funding-sources-50`,
                ),
              )
            }
          />
          <div className=" text-[14px] sm:text-[16px]">
            {isShowingFundingSources ? (
              <div className="flex items-center gap-x-1 ">
                {block(() => {
                  const w = layout === 'mobile' ? 37 : 45
                  return (
                    <div
                      className="relative rounded-full text-[13px] sm:text-[14px] overflow-hidden"
                      style={{
                        backgroundColor: planColors.shades.light[3].hex,
                      }}
                    >
                      <div
                        className="absolute h-full z-0"
                        style={{
                          transitionProperty: 'transform',
                          transitionDuration: '300ms',
                          transform: `translateX(${
                            PERCENTILES_STR.indexOf(currPrecentile) * w
                          }px)`,
                          width: `${w}px`,
                          backgroundColor: planColors.shades.light[10].hex,
                        }}
                      />
                      {PERCENTILES_STR.map((percentileStr, i) => (
                        <button
                          key={percentileStr}
                          className={clix(
                            'relative z-10',
                            i !== 0 && ' border-l',
                          )}
                          style={{
                            width: `${w}px`,
                            transitionProperty: 'color',
                            transitionDuration: '300ms',
                            // color:
                            //   percentileStr === currPrecentile
                            //     ? planColors.fgForDarkBG.hex
                            //     : undefined,
                            borderColor: planColors.shades.light[10].hex,
                          }}
                          onClick={() => {
                            urlUpdater.push(
                              getPlanChartURL(
                                `spending-total-funding-sources-${percentileStr}`,
                              ),
                            )
                          }}
                        >
                          {percentileStr}
                          <span className=" align-super text-[8px] sm:text-[10px]">
                            th
                          </span>
                        </button>
                      ))}
                    </div>
                  )
                })}
                percentile
                {/* <div className="">
                <button className={'pl-2 pr-2 '} onClick={() => handleDelta(1)}>
                  <FontAwesomeIcon icon={faPlus} />
                </button>
                <button
                  className={'pr-2 pl-2 '}
                  onClick={() => handleDelta(-1)}
                >
                  <FontAwesomeIcon icon={faMinus} />
                </button>
              </div> */}
              </div>
            ) : (
              <div className="lighten">Show funding sources</div>
            )}
          </div>
        </div>
      </div>
    )
  },
)
