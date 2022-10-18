import {faCaretDown, faCheck, faSave} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import isMobile from 'is-mobile'
import _ from 'lodash'
import React, {useEffect, useLayoutEffect, useRef, useState} from 'react'
import ReactDOM from 'react-dom'
import {resolveTPAWRiskPreset} from '../../../TPAWSimulator/DefaultParams'
import {riskLevelLabel} from '../../../TPAWSimulator/RiskLevelLabel'
import {TPAWRiskLevel} from '../../../TPAWSimulator/TPAWParams'
import {getNumYears} from '../../../TPAWSimulator/TPAWParamsExt'
import {applyOriginToHTMLElement} from '../../../Utils/Geometry'
import {assert, fGet} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {useWindowSize} from '../../App/WithWindowSize'

const duration = 500
const scale = 0.95

export const PlanSummaryRiskCopyFromAPreset = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, setParams} = useSimulation()
    const windowSize = useWindowSize()

    const referenceElementRef = useRef<HTMLButtonElement | null>(null)
    const popperElementRef = useRef<HTMLDivElement | null>(null)
    const [height, setHeight] = useState(0)
    const [show, setShow] = useState(false)

    useLayoutEffect(() => {
      const observer = new ResizeObserver(() =>
        setHeight(
          fGet(popperElementRef.current).getBoundingClientRect().height / scale
        )
      )
      observer.observe(fGet(popperElementRef.current))
      return () => observer.disconnect()
    }, [])

    const handleShow = () => {
      setShow(true)
      const position = fGet(referenceElementRef.current).getBoundingClientRect()
      const origin = {
        y:  Math.min(position.top, windowSize.height - height - 20),
        x: isMobile() ? 0 : position.left,
      }
      applyOriginToHTMLElement(origin, fGet(popperElementRef.current))
    }
    const [opacity0AtTransitionEnd, setOpacity0AtTransitionEnd] = useState(true)
    const invisible = !show && opacity0AtTransitionEnd

    const handleClick = (preset: TPAWRiskLevel | 'saved') => {
      setShow(false)
      setParams(params => {
        const clone = _.cloneDeep(params)
        assert(!clone.risk.useTPAWPreset)
        const tpawRisk = _.cloneDeep(
          preset === 'saved'
            ? fGet(clone.risk.savedTPAWPreset)
            : resolveTPAWRiskPreset(
                {...clone.risk, useTPAWPreset: true, tpawPreset: preset},
                getNumYears(clone)
              )
        )
        clone.risk = {...clone.risk, ...tpawRisk}
        return clone
      })
    }

    const [markAsSaved, setMarkAsSaved] = useState(false)
    useEffect(() => {
      const timeout = window.setTimeout(() => setMarkAsSaved(false), 750)
      return () => window.clearTimeout(timeout)
    }, [markAsSaved])

    return (
      <>
        <button
          ref={referenceElementRef}
          className={`${className} flex items-center gap-x-2`}
          onClick={handleShow}
        >
          Copy from a Preset
          <FontAwesomeIcon className="text-lg" icon={faCaretDown} />
        </button>
        {ReactDOM.createPortal(
          <div
            className=" page fixed inset-0"
            style={{
              visibility: invisible ? 'hidden' : 'visible',
              transitionProperty: 'opacity',
              transitionDuration: `${duration}ms`,
              opacity: show ? '1' : '0',
            }}
            onTransitionEnd={() => setOpacity0AtTransitionEnd(!show)}
          >
            <div
              className="fixed inset-0 bg-black opacity-70"
              onClick={() => setShow(false)}
            />
            <div
              className={`flex absolute flex-col  rounded-xl   bg-planBG`}
              ref={popperElementRef}
              style={{
                transitionProperty: 'transform',
                transitionDuration: `${duration}ms`,
                transform: `scale(${show ? 1 : scale})`,
                width:
                  windowSize.width < 600 ? `${windowSize.width}px` : undefined,
                boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
              }}
            >
              <div className="flex flex-col mt-4   ">
                <h2 className="font-bold mx-4 text-xl">Presets</h2>
                <button
                  className="text-left py-2 px-4 font-medium"
                  onClick={() => handleClick('riskLevel-1')}
                >
                  {riskLevelLabel('riskLevel-1')}
                </button>
                <button
                  className="text-left py-2 px-4 font-medium "
                  onClick={() => handleClick('riskLevel-2')}
                >
                  {riskLevelLabel('riskLevel-2')}
                </button>
                <button
                  className="text-left py-2 px-4 font-medium "
                  onClick={() => handleClick('riskLevel-3')}
                >
                  {riskLevelLabel('riskLevel-3')}
                </button>
                <button
                  className="text-left py-2 px-4 font-medium "
                  onClick={() => handleClick('riskLevel-4')}
                >
                  {riskLevelLabel('riskLevel-4')}
                </button>
              </div>
              <div className=" px-4 text-left font-medium mt-4">
                <h2 className="font-bold text-lg">Custom</h2>
                <button
                  className=" disabled:lighten-2 py-2"
                  disabled={params.risk.savedTPAWPreset === null}
                  onClick={() => handleClick('saved')}
                >
                  {markAsSaved ? (
                    <span>
                      <FontAwesomeIcon icon={faCheck} /> Saved
                    </span>
                  ) : (
                    'Last Saved Risk Profile'
                  )}
                </button>
                <div className="flex justify-end mt-1">
                  <button
                    className="flex items-center gap-x-2 py-2 mb-2  btn-sm rounded-full bg-gray-300 "
                    onClick={() => {
                      setMarkAsSaved(true)
                      setParams(params => {
                        assert(!params.risk.useTPAWPreset)
                        const clone = _.cloneDeep(params)
                        clone.risk.savedTPAWPreset = _.cloneDeep({
                          tpaw: params.risk.tpaw,
                          tpawAndSPAW: params.risk.tpawAndSPAW,
                        })
                        return clone
                      })
                    }}
                  >
                    <FontAwesomeIcon icon={faSave} />
                    Save Now
                  </button>
                </div>
              </div>
            </div>
          </div>,
          window.document.body
        )}
      </>
    )
  }
)
