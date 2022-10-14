import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from 'react'
import {
  applyOriginToHTMLElement,
  Size,
  sizeCSSStyle,
  XY,
} from '../../Utils/Geometry'
import {useAssertConst} from '../../Utils/UseAssertConst'
import {fGet} from '../../Utils/Utils'
import {useSimulation} from '../App/WithSimulation'
import {PlanInputType} from './PlanInput/Helpers/PlanInputType'
import {planSectionLabel} from './PlanInput/Helpers/PlanSectionLabel'

export type PlanHeadingStateful = {
  setTransition: (transition: number) => void
}

type Props = {
  type: PlanInputType
  sizing: {
    dynamic: (transition: number) => {origin: XY}
    fixed: {size: Size}
  }
  transitionRef: React.MutableRefObject<{transition: number}>
  onDone: () => void
}

export type PlanHeadingSizing = Props['sizing']

export const PlanHeading = React.memo(
  React.forwardRef<PlanHeadingStateful, Props>(
    ({type, sizing, transitionRef, onDone}: Props, forwardRef) => {
      const {params} = useSimulation()
      const outerRef = useRef<HTMLDivElement | null>(null)
      const setTransition = useCallback(
        (transition: number) => {
          const {origin} = sizing.dynamic(transition)
          const outer = fGet(outerRef.current)
          outer.style.opacity = `${transition}`
          outer.style.display = transition === 0 ? 'none' : 'flex'
          applyOriginToHTMLElement(origin, outer)
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      const {size} = sizing.fixed
      return (
        <div
          className="absolute  z-10 items-center bg-planBG bg-opacity-90 "
          ref={outerRef}
          style={{...sizeCSSStyle(size)}}
        >
          {/* <div className="flex  items-center gap-x-4  bg-planBG bg-opacity-90 rounded-br-xl">
            <button
              className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5"
              onClick={onDone}
            >
              <FontAwesomeIcon className="" icon={faLeftLong} />
              Done
            </button>
            <h2 className="text-xl sm:text-2xl font-bold text-start pr-5">
              {planSectionLabel(type)}
            </h2>
          </div> */}
          <div className="w-full flex items-center gap-x-2">
            <div className="w-[50px] h-[50px] bg-gray-700 rounded-full flex items-center justify-center text-white text-xl">
              1
            </div>
            <h2 className="font-bold text-3xl">
              {planSectionLabel(type, params.strategy)}
            </h2>
          </div>
        </div>
      )
    }
  )
)
