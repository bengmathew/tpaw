import {
  faCaretDown,
  faChevronRight,
  faInfo,
} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React from 'react'
import {useSimulation} from '../../../App/WithSimulation'
import {useGetSectionURL} from '../../Plan'
import {PlanChartInternalTransitionState} from '../PlanChart'
import {usePlanChartType} from '../UsePlanChartType'
import {planChartLabel} from './PlanChartLabel'

type Props = {
  className?: string
  onClick: () => void
  style?: React.CSSProperties
  layout: 'mobile' | 'desktop' | 'laptop'
  transition: {target: PlanChartInternalTransitionState; duration: number}
}
export const PlanChartMainCardMenuButton = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    ({className = '', onClick, style, layout, transition}: Props, ref) => {
      const simulation = useSimulation()
      const {params} = simulation.tpawResult.args
      const getSectionURL = useGetSectionURL()
      const type = usePlanChartType()
      const {label, subLabel} = planChartLabel(params.original, type, 'full')

      const showHelp =
      transition.target === 'summaryWithLegacy' ||
      transition.target === 'summaryWithoutLegacy'
      
      return (
        <div
          className={`${className} flex justify-between items-start gap-x-1`}
          ref={ref}
        >
          <button
            className={`flex items-start gap-x-2 text-[16px] sm:text-[22px] font-bold text-left`}
            onClick={onClick}
            style={{...style}}
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
          {layout === 'mobile' && (
            <Link href={getSectionURL('results')} shallow>
              <a
                className="flex items-center justify-center shrink-0 w-[25px] h-[25px] bg-gray-200 rounded-full"
                style={{
                  transitionProperty: 'opacity',
                  transitionDuration: `${transition.duration}ms`,
                  opacity: showHelp ? '1' : '0',
                  pointerEvents: showHelp ? 'auto' : 'none',
                }}
              >
                <FontAwesomeIcon icon={faInfo} />
              </a>
            </Link>
          )}
        </div>
      )
    }
  )
)
