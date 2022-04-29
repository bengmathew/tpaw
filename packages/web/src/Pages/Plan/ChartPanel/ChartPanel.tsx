import {faArrowsUpDown, faClipboardList} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, {
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  applyPaddingToHTMLElement,
  applyPositionToHTMLElement,
  applyRectSizingToHTMLElement,
  Padding,
  rectExt,
  RectExt,
} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet} from '../../../Utils/Utils'
import {ChartRef} from '../../Common/Chart/ChartReact'
import {ChartPanelDescription} from './ChartPanelDescription'
import {ChartPanelMenu, ChartPanelMenuStateful} from './ChartPanelMenu'
import {ChartPanelMenuButton} from './ChartPanelMenuButton'
import {TPAWChart} from './TPAWChart'
import {ChartPanelState} from './UseChartPanelState'

export type ChartPanelStateful = {
  setTransition: (transition: number) => void
}

type Props = {
  layout: 'laptop' | 'desktop' | 'mobile'
  sizing: (transition: number) => {
    position: RectExt
    padding: Padding
    cardPadding: Padding
    headingMarginBottom: number
    menuButtonScale: number
  }
  transitionRef: React.MutableRefObject<{
    transition: number
  }>
  state: ChartPanelState
}

export const ChartPanel = React.memo(
  React.forwardRef<ChartPanelStateful, Props>(
    (
      {
        state: {handleChangeType, state, yAxisFormat, handleRescale},
        layout,
        sizing: sizingIn,
        transitionRef,
      }: Props,
      forwardRef
    ) => {
      const [showDescriptionPopUp, setShowDescriptionPopUp] = useState(false)
      const outerRef = useRef<HTMLDivElement | null>(null)
      const headingRef = useRef<HTMLHeadingElement | null>(null)
      const bodyRef = useRef<HTMLDivElement | null>(null)
      const menuRef = useRef<ChartPanelMenuStateful | null>(null)
      const at0WidthDivRef = useRef<HTMLDivElement | null>(null)
      const at1WidthDivRef = useRef<HTMLDivElement | null>(null)
      const descriptionRef = useRef<HTMLDivElement | null>(null)
      const chartRef = useRef<ChartRef | null>(null)
      const tasksRef = useRef<HTMLAnchorElement | null>(null)
      const rescaleRef = useRef<HTMLButtonElement | null>(null)

      const [measures, setMeasures] = useState({
        heading: 0,
        top: {at0: 0, at1: 0},
        tasks: 0,
      })
      useLayoutEffect(() => {
        const observer = new ResizeObserver(() => {
          setMeasures({
            heading: headingRef.current?.getBoundingClientRect().height ?? 0,
            top: {
              at0: fGet(at0WidthDivRef.current).getBoundingClientRect().height,
              at1: fGet(at1WidthDivRef.current).getBoundingClientRect().height,
            },
            tasks: fGet(tasksRef.current).getBoundingClientRect().height,
          })
        })
        if (headingRef.current) {
          observer.observe(headingRef.current)
        }
        observer.observe(fGet(at0WidthDivRef.current))
        observer.observe(fGet(at1WidthDivRef.current))
        observer.observe(fGet(rescaleRef.current))
        return () => observer.disconnect()
      }, [])

      const bodyContentWidth = useCallback(
        (transition: number) => {
          const {position, padding, cardPadding} = sizingIn(transition)
          return (
            position.width -
            padding.left -
            padding.right -
            cardPadding.left -
            cardPadding.right
          )
        },
        [sizingIn]
      )

      const sizing = useCallback(
        (transition: number) => {
          const orig = sizingIn(transition)
          const {position, cardPadding, padding, headingMarginBottom} = orig

          const bodyW = position.width - padding.left - padding.right
          const bodyY = padding.top + measures.heading + headingMarginBottom
          const bodyH =
            position.height -
            bodyY -
            measures.tasks -
            padding.bottom -
            padding.bottom

          const bodyPosition = rectExt({
            height: bodyH,
            width: bodyW,
            x: padding.left,
            y: bodyY,
          })
          const top = linearFnFomPoints(
            0,
            measures.top.at0,
            1,
            measures.top.at1
          )(transition)
          const chartPosition = rectExt({
            height:
              bodyPosition.height - top - cardPadding.top - cardPadding.bottom,
            x: 0,
            y: top + cardPadding.top,
            width: bodyPosition.width,
          })

          const rescalePosition = {
            x: padding.left + 30,
            y: bodyPosition.bottom - 10,
            height: 30,
          }

          const tasksPosition = {
            right: padding.right,
            // left: bodyPosition.x,
            y: bodyPosition.bottom + padding.bottom,
          }

          const descriptionScale = linearFnFomPoints(
            0,
            1,
            1,
            bodyContentWidth(1) / bodyContentWidth(0)
          )(transition)

          return {
            ...orig,
            bodyPosition,
            chartPosition,
            rescalePosition,
            tasksPosition,
            descriptionScale,
          }
        },
        [sizingIn, measures, bodyContentWidth]
      )

      const setTransition = useCallback(
        (transition: number) => {
          const {
            position,
            padding,
            cardPadding,
            headingMarginBottom,
            bodyPosition,
            chartPosition,
            menuButtonScale,
            rescalePosition,
            tasksPosition,
            descriptionScale,
          } = sizing(transition)

          applyRectSizingToHTMLElement(position, fGet(outerRef.current))
          applyPaddingToHTMLElement(padding, fGet(outerRef.current))
          if (headingRef.current) {
            headingRef.current.style.marginBottom = `${headingMarginBottom}px`
            headingRef.current.style.paddingLeft = `${linearFnFomPoints(
              0,
              sizing(0).cardPadding.left,
              1,
              sizing(1).cardPadding.left
            )(transition)}px`
          }

          applyRectSizingToHTMLElement(bodyPosition, fGet(bodyRef.current))
          applyPaddingToHTMLElement(cardPadding, fGet(bodyRef.current))

          const transitionReverse = linearFnFomPoints(0, 1, 1, 0)(transition)

          fGet(menuRef.current).setButtonScale(menuButtonScale)
          fGet(descriptionRef.current).style.opacity = `${transitionReverse}`
          fGet(descriptionRef.current).style.transform = `scale(${
            bodyContentWidth(transition) / bodyContentWidth(0)
          })`

          fGet(chartRef.current).setPosition(chartPosition)

          applyPositionToHTMLElement(rescalePosition, fGet(rescaleRef.current))
          fGet(rescaleRef.current).style.height = `${rescalePosition.height}px`

          fGet(tasksRef.current).style.right = `${tasksPosition.right}px`
          fGet(tasksRef.current).style.top = `${tasksPosition.y}px`

          fGet(tasksRef.current).style.opacity = `${transitionReverse}`
          fGet(tasksRef.current).style.visibility =
            transition === 1 ? 'hidden' : 'visible'
        },
        [bodyContentWidth, sizing]
      )
      useImperativeHandle(forwardRef, () => ({setTransition}), [setTransition])
      useLayoutEffect(() => {
        setTransition(transitionRef.current.transition)
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      return (
        <div
          className={`absolute top-0  right-0 z-10 bg-gray-300 
          ${layout == 'laptop' ? 'rounded-2xl' : ''}`}
          ref={outerRef}
        >
          {layout !== 'mobile' && (
            <h2 className={`absolute uppercase font-bold `} ref={headingRef}>
              Results
            </h2>
          )}
          <div
            className="absolute bg-cardBG   rounded-t-2xl rounded-b-2xl overflow-hidden"
            ref={bodyRef}
          >
            <div className="">
              <ChartPanelMenu
                type={state.type}
                onSelect={handleChangeType}
                ref={menuRef}
              />
            </div>
            <div
              className=""
              ref={descriptionRef}
              style={{
                width: `${bodyContentWidth(0)}px`,
                transformOrigin: 'top left',
              }}
            >
              <ChartPanelDescription
                type={state.type}
                showDescriptionPopUp={showDescriptionPopUp}
                setShowDescriptionPopUp={setShowDescriptionPopUp}
              />
            </div>
            <div
              className="absolute invisible "
              style={{width: `${bodyContentWidth(0)}px`}}
              ref={at0WidthDivRef}
            >
              <ChartPanelMenuButton
                type={state.type}
                onClick={() => {}}
                style={{scale: `${sizingIn(0).menuButtonScale}`}}
              />
              <ChartPanelDescription
                type={state.type}
                showDescriptionPopUp={false}
                setShowDescriptionPopUp={() => {}}
              />
            </div>
            <div
              className="absolute invisible"
              style={{width: `${bodyContentWidth(1)}px`}}
              ref={at1WidthDivRef}
            >
              <ChartPanelMenuButton
                type={state.type}
                onClick={() => {}}
                style={{scale: `${sizingIn(1).menuButtonScale}`}}
              />
            </div>
            <TPAWChart
              yAxisFormat={yAxisFormat}
              state={state}
              startingPosition={
                sizing(transitionRef.current.transition).chartPosition
              }
              chartRef={x => (chartRef.current = x)}
              extraPaddingTop={layout === 'mobile' ? 0 : 15}
            />
          </div>
          <button
            className="absolute bg-gray-100 border border-gray-400 rounded-full px-2 py-1 flex gap-x-2 items-center font-bol"
            onClick={handleRescale}
            ref={rescaleRef}
          >
            <FontAwesomeIcon className="" icon={faArrowsUpDown} />
            Rescale
          </button>
          <Link href="/tasks-for-this-year">
            <a
              className="absolute whitespace-nowrap bg-cardB py-1 rounded-xl flex items-center gap-x-2"
              ref={tasksRef}
            >
              <h2 className="font-bold text-base sm:text-xl flex items-center gap-x-2">
                <FontAwesomeIcon icon={faClipboardList} />
                Tasks for this year{' '}
                {/* <FontAwesomeIcon icon={faLongArrowAltRight} /> */}
              </h2>
            </a>
          </Link>
        </div>
      )
    }
  )
)
