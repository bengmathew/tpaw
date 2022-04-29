import {faCaretDown, faChevronRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React from 'react'
import {useSimulation} from '../../App/WithSimulation'
import {chartPanelLabel} from './ChartPanelLabel'
import {ChartPanelType} from './ChartPanelType'

type Props = {
  className?: string
  onClick: () => void
  type: ChartPanelType
  style?: React.CSSProperties
}
export const ChartPanelMenuButton = React.memo(
  React.forwardRef<HTMLButtonElement, Props>(
    ({className = '', onClick, type, style}: Props, ref) => {
      const simulation = useSimulation()
      const {params} = simulation.tpawResult.args
      const {label, subLabel} = chartPanelLabel(params, type, 'full')
      return (
        <button
          className={`${className}  flex items-start gap-x-2 text-lg sm:text-2xl  font-bold  text-left"`}
          onClick={onClick}
          ref={ref}
          style={{...style, transformOrigin:'top left'}}
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
      )
    }
  )
)
