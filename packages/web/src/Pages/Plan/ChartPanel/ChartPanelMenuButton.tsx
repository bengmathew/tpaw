import { faInfo } from '@fortawesome/free-solid-svg-icons'
import {faCaretDown, faChevronRight, faInfoCircle} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React from 'react'
import {useSimulation} from '../../App/WithSimulation'
import {chartPanelLabel} from './ChartPanelLabel'
import {ChartPanelType} from './ChartPanelType'

type Props = {
  layout: 'mobile' | 'desktop' | 'laptop'
  className?: string
  onClick: () => void
  type: ChartPanelType | 'sharpe-ratio'
  showDescriptionPopUp: () => void
  style?: React.CSSProperties
}
export const ChartPanelMenuButton = React.memo(
  React.forwardRef<HTMLDivElement, Props>(
    (
      {
        className = '',
        onClick,
        type,
        style,
        layout,
        showDescriptionPopUp,
      }: Props,
      ref
    ) => {
      const simulation = useSimulation()
      const {params} = simulation.tpawResult.args
      const {label, subLabel} = chartPanelLabel(params.original, type, 'full')
      return (
        <div
          className="flex justify-between gap-x-4 items-start"
          ref={ref}
          style={{...style, transformOrigin: 'top left'}}
        >
          <button
            className={`${className}  flex items-start gap-x-2 text-[20px] sm:text-[22px] font-bold text-left`}
            onClick={onClick}
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
            <button
              className=" w-[22px] h-[22px] mt-1  text-black bg-gray-300 rounded-full flex items-center justify-center text-sm shrink-0 pb-0.5"
              onClick={showDescriptionPopUp}
            >
              <FontAwesomeIcon icon={faInfo}/>
            </button>
          )}
        </div>
      )
    }
  )
)
