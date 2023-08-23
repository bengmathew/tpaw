import { faCaretDown, faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React from 'react'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanResultsTransitionState } from '../PlanResults'
import { usePlanResultsChartType } from '../UsePlanResultsChartType'
import { planResultsChartLabel } from './PlanResultsChartLabel'

type Props = {
  className?: string
  onClick: () => void
  style?: React.CSSProperties
  transition: { target: PlanResultsTransitionState; duration: number }
}
export const PlanResultsChartCardMenuButton = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    ({ className = '', onClick, style, transition }: Props, ref) => {
      const simulation = useSimulation()
      const { params } = simulation.tpawResult
      const type = usePlanResultsChartType()
      const { label, subLabel } = planResultsChartLabel(
        params.original,
        type,
        'full',
      )

      return (
        <div
          className={`${className} flex justify-between items-start gap-x-1`}
          ref={ref}
        >
          <button
            className={`flex items-start gap-x-2 text-[16px] sm:text-[22px] font-bold text-left`}
            onClick={onClick}
            style={{ ...style }}
          >
            <div className="">
              <h2 className="">
                {label.map((x, i) => (
                  <React.Fragment key={i}>
                    <span>{x}</span>
                    {i !== label.length - 1 && (
                      <FontAwesomeIcon
                        className="mx-1.5 text-sm sm:text-base lighten-2"
                        icon={faChevronRight}
                      />
                    )}
                  </React.Fragment>
                ))}
              </h2>
              {subLabel && (
                <h2 className="font-bold text-base sm:text-xl">{subLabel}</h2>
              )}
            </div>
            <FontAwesomeIcon className="mt-1" icon={faCaretDown} />
          </button>
        </div>
      )
    },
  ),
)
