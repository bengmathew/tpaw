import { faCaretDown, faChevronRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React from 'react'
import { useSimulation } from '../../../App/WithSimulation'
import { useGetSectionURL } from '../../Plan'
import { PlanChartTransitionState } from '../PlanChart'
import { usePlanChartType } from '../UsePlanChartType'
import { planChartLabel } from './PlanChartLabel'

type Props = {
  className?: string
  onClick: () => void
  style?: React.CSSProperties
  transition: { target: PlanChartTransitionState; duration: number }
}
export const PlanChartMainCardMenuButton = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    ({ className = '', onClick, style, transition }: Props, ref) => {
      const simulation = useSimulation()
      const { params } = simulation.tpawResult.args
      const getSectionURL = useGetSectionURL()
      const type = usePlanChartType()
      const { label, subLabel } = planChartLabel(params.original, type, 'full')

      const showHelp = transition.target === 'summary'

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
