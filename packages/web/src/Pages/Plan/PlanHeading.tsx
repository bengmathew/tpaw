import {faLeftLong} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from 'react'
import {applyRectSizingToHTMLElement, RectExt} from '../../Utils/Geometry'
import {useAssertConst} from '../../Utils/UseAssertConst'
import {fGet} from '../../Utils/Utils'
import {paramsInputLabel} from './ParamsInput/Helpers/ParamsInputLabel'
import {ParamsInputType} from './ParamsInput/Helpers/ParamsInputType'

export type PlanHeadingStateful = {
  setTransition: (transition: number) => void
}

type Props = {
  type: ParamsInputType
  sizing: (transition: number) => {position: RectExt}
  transitionRef: React.MutableRefObject<{transition: number}>
  onDone: () => void
}

export const PlanHeading = React.memo(
  React.forwardRef<PlanHeadingStateful, Props>(
    ({type, sizing, transitionRef, onDone}: Props, forwardRef) => {
      const outerRef = useRef<HTMLDivElement | null>(null)
      const setTransition = useCallback(
        (transition: number) => {
          const {position} = sizing(transition)
          const outer = fGet(outerRef.current)
          outer.style.opacity = `${transition}`
          outer.style.display = transition === 0 ? 'none' : 'flex'
          applyRectSizingToHTMLElement(position, outer)
        },
        [sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      return (
        <div className="absolute  z-10" ref={outerRef}>
          <div className="flex w-full items-center gap-x-4  bg-planBG bg-opacity-80">
            <button
              className="flex items-center gap-x-2 text-sm sm:text-base btn-dark px-4 py-1.5"
              onClick={onDone}
            >
              <FontAwesomeIcon className="" icon={faLeftLong} />
              Done
            </button>
            <h2 className="text-xl sm:text-2xl font-bold text-start">
              {paramsInputLabel(type)}
            </h2>
          </div>
        </div>
      )
    }
  )
)
