import { faCaretDown, faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React from 'react'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanResultsTransitionState } from '../PlanResults'
import { usePlanResultsChartType } from '../UsePlanResultsChartType'
import { planResultsChartLabel } from './PlanResultsChartLabel'

type Props = {
  className?: string
  style?: React.CSSProperties
  transition: { target: PlanResultsTransitionState; duration: number }
}
export const PlanResultsChartCardMenuButton = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    ({ className = '', style }: Props, ref) => {
      const { simulationResult } = useSimulation()
      const { planParamsNorm } = simulationResult.args
      const type = usePlanResultsChartType()
      const { label, subLabel } = planResultsChartLabel(planParamsNorm, type)

      return (
        <div
          className={`${className} flex justify-between items-start gap-x-1`}
          ref={ref}
        >
          <div
            className={`flex items-start gap-x-2 text-[16px] sm:text-[22px] font-bold text-left`}
            style={{ ...style }}
          >
            <div className="">
              <h2 className="">
                {label.full.map((x, i) => (
                  <React.Fragment key={i}>
                    <span>{x}</span>
                    {i !== label.full.length - 1 && (
                      <FontAwesomeIcon
                        className="mx-1.5 text-sm sm:text-base lighten-2"
                        icon={faChevronRight}
                      />
                    )}
                  </React.Fragment>
                ))}
              </h2>
              {subLabel && (
                <h2 className="font-bold text-base sm:text-xl ">{subLabel}</h2>
              )}
            </div>
            <FontAwesomeIcon className="mt-1" icon={faCaretDown} />
          </div>
        </div>
      )
    },
  ),
)
