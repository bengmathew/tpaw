import {faArrowsUpDown} from '@fortawesome/pro-light-svg-icons'
import {faClipboardList} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import gsap from 'gsap'
import Link from 'next/link'
import React, {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
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
import {SimpleRange} from '../../../Utils/SimpleRange'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {fGet} from '../../../Utils/Utils'
import {ChartReactStatefull} from '../../Common/Chart/ChartReact'
import {ChartPanelDescription} from './ChartPanelDescription'
import {ChartPanelMenu, ChartPanelMenuStateful} from './ChartPanelMenu'
import {ChartPanelMenuButton} from './ChartPanelMenuButton'
import {TPAWChartLegacy} from './TPAWChart/TPAWChartLegacy'
import {TPAWChartMain} from './TPAWChart/TPAWChartMain'
import {ChartPanelState} from './UseChartPanelState'

export type ChartPanelStateful = {
  setTransition: (transition: number, target: 0 | 1) => void
}

type Props = {
  layout: 'laptop' | 'desktop' | 'mobile'
  sizing: (transition: number) => {
    position: RectExt
    padding: Padding
    cardPadding: Padding
    headingMarginBottom: number
    menuButtonScale: number
    legacyWidth: number
    intraGap: number
  }
  transitionRef: React.MutableRefObject<{
    transition: number
    target: 0 | 1
  }>
  state: ChartPanelState
}

export const ChartPanel = React.memo(
  React.forwardRef<ChartPanelStateful, Props>(
    (
      {
        state: {
          handleChangeType,
          state,
          handleRescale,
          shouldShowLegacy,
          targetYRange,
        },
        layout,
        sizing: sizingIn,
        transitionRef,
      }: Props,
      forwardRef
    ) => {
      const rescaleWarningLevel = _maxRescaleWarningLevel(
        _rescaleWarningLevel(state.main.xyRange.y, targetYRange.main),
        shouldShowLegacy
          ? _rescaleWarningLevel(state.legacy.xyRange.y, targetYRange.legacy)
          : 0
      )
      const [showDescriptionPopUp, setShowDescriptionPopUp] = useState(false)
      const outerRef = useRef<HTMLDivElement | null>(null)
      const headingRef = useRef<HTMLHeadingElement | null>(null)
      const bodyRef = useRef<HTMLDivElement | null>(null)
      const menuDivRef = useRef<HTMLDivElement | null>(null)
      const menuRef = useRef<ChartPanelMenuStateful | null>(null)
      const menuAt0DivRef = useRef<HTMLDivElement | null>(null)
      const menuAt1DivRef = useRef<HTMLDivElement | null>(null)
      const descriptionRef = useRef<HTMLDivElement | null>(null)
      const descriptionAt0Ref = useRef<HTMLDivElement | null>(null)
      const mainChartRef = useRef<ChartReactStatefull | null>(null)
      const legacyBodyRef = useRef<HTMLDivElement | null>(null)
      const legacyHeadingRef = useRef<HTMLHeadingElement | null>(null)
      const legacyChartRef = useRef<ChartReactStatefull | null>(null)
      const tasksRef = useRef<HTMLAnchorElement | null>(null)
      const rescaleRef = useRef<HTMLButtonElement | null>(null)
      const legacyTransitionRef = useRef({transition: shouldShowLegacy ? 1 : 0})

      const [measures, setMeasures] = useState({
        heading: 0,
        menu: {at0: 0, at1: 0},
        descriptionAt0: 0,
        tasks: 0,
      })
      useLayoutEffect(() => {
        const observer = new ResizeObserver(() => {
          setMeasures({
            heading: headingRef.current?.getBoundingClientRect().height ?? 0,
            menu: {
              at0: fGet(menuAt0DivRef.current).getBoundingClientRect().height,
              at1: fGet(menuAt1DivRef.current).getBoundingClientRect().height,
            },
            descriptionAt0: fGet(
              descriptionAt0Ref.current
            ).getBoundingClientRect().height,
            tasks: fGet(tasksRef.current).getBoundingClientRect().height,
          })
        })
        if (headingRef.current) {
          observer.observe(headingRef.current)
        }
        observer.observe(fGet(menuAt0DivRef.current))
        observer.observe(fGet(menuAt1DivRef.current))
        observer.observe(fGet(descriptionAt0Ref.current))
        observer.observe(fGet(rescaleRef.current))
        return () => observer.disconnect()
      }, [])

      const sizing = useCallback(
        (transition: number, target: 0 | 1, legacyTransition: number) => {
          const orig = sizingIn(transition)
          const origAt0 = sizingIn(0)
          const origAt1 = sizingIn(1)
          const origAtTarget = target === 0 ? origAt0 : origAt1
          const {
            position,
            cardPadding,
            padding,
            headingMarginBottom,
            legacyWidth,
            menuButtonScale,
          } = orig

          const bodyY = padding.top + measures.heading + headingMarginBottom
          const bodyH =
            position.height - bodyY - measures.tasks - padding.bottom

          const headingPaddingLeft = linearFnFomPoints(
            0,
            origAt0.cardPadding.left,
            1,
            origAt1.cardPadding.left
          )(transition)

          const legacyBodyPosition = rectExt({
            height: bodyH,
            width: legacyWidth,
            x: legacyBodyX(orig, legacyTransition),
            y: bodyY,
          })

          const legacyPosition = rectExt({
            height: legacyBodyPosition.height,
            y: 0,
            x: 0,
            width: legacyBodyPosition.width,
          })

          const bodyPosition = rectExt({
            height: bodyH,
            width: bodyWFn(orig, legacyTransition),
            x: padding.left,
            y: bodyY,
          })

          const top = linearFnFomPoints(
            0,
            measures.menu.at0 + measures.descriptionAt0,
            1,
            measures.menu.at1
          )(transition)

          const legacyPadding = {left: 0, right: 0, top: 60, bottom: 10}

          const legacySizing = {
            position: legacyPosition,
            padding: legacyPadding,
          }

          const legacyTarget = shouldShowLegacy ? 1 : 0
          const menuPosition = {
            x: cardPadding.left,
            y: cardPadding.top,
            width: bodyContentWFn(origAtTarget, legacyTarget),
          }
          const descriptionPosition = {
            x: cardPadding.left,
            y: menuPosition.y + measures.menu.at0 * menuButtonScale,
            width: bodyContentWFn(origAt0, legacyTarget),
          }
          const chartPosition = rectExt({
            height:
              bodyPosition.height - top - cardPadding.top - cardPadding.bottom,
            x: 0,
            y: top + cardPadding.top,
            width: bodyPosition.width,
          })

          const chartPadding = {
            left: cardPadding.left,
            right: cardPadding.right,
            top: layout === 'mobile' ? 20 : 30,
            bottom: 35,
          }
          const chartSizing = {position: chartPosition, padding: chartPadding}

          const rescalePosition = {
            x: padding.left,
            y:
              bodyPosition.bottom +
              (position.height - bodyPosition.bottom - 25) / 2,
            height: 25,
          }

          const tasksPosition = {
            right: padding.right,
            // y: bodyPosition.bottom ,

            y:
              bodyPosition.bottom +
              (position.height - bodyPosition.bottom - measures.tasks) / 2,
            // y: bodyPosition.bottom + ((padding.bottom-bodyPosition.bottom) - 30)/2 ,
          }

          const descriptionScale = linearFnFomPoints(
            0,
            1,
            1,
            // This is approximate, but that is fine because it is hidden when
            // transition is 1.
            origAt1.position.width / origAt0.position.width
          )(transition)

          return {
            ...orig,
            bodyPosition,
            headingPaddingLeft,
            rescalePosition,
            tasksPosition,
            descriptionScale,
            chartSizing,
            legacyBodyPosition,
            legacySizing,
            menuPosition,
            descriptionPosition,
          }
        },
        [sizingIn, measures, shouldShowLegacy, layout]
      )

      const [startingSizing] = useState(() =>
        sizing(
          transitionRef.current.transition,
          transitionRef.current.target,
          legacyTransitionRef.current.transition
        )
      )
      const sizingAt01 = useMemo(() => sizing(0, 0, 1), [sizing])
      const sizingAt00 = useMemo(() => sizing(0, 0, 0), [sizing])
      const sizingAt11 = useMemo(() => sizing(1, 1, 1), [sizing])
      const sizingAt10 = useMemo(() => sizing(1, 1, 0), [sizing])
      const sizingAt0AtTargetLegacy = shouldShowLegacy ? sizingAt01 : sizingAt00
      const sizingAt1AtTargetLegacy = shouldShowLegacy ? sizingAt11 : sizingAt10

      const setTransition = useCallback(
        (transition: number, target: 0 | 1, legacyTransition: number) => {
          const {
            position,
            padding,
            cardPadding,
            headingMarginBottom,
            bodyPosition,
            chartSizing,
            legacyBodyPosition,
            legacySizing,
            menuButtonScale,
            rescalePosition,
            tasksPosition,
            descriptionScale,
            headingPaddingLeft,
            menuPosition,
            descriptionPosition,
          } = sizing(transition, target, legacyTransition)

          applyRectSizingToHTMLElement(position, fGet(outerRef.current))
          applyPaddingToHTMLElement(padding, fGet(outerRef.current))
          if (headingRef.current) {
            headingRef.current.style.marginBottom = `${headingMarginBottom}px`
            headingRef.current.style.paddingLeft = `${headingPaddingLeft}px`
          }

          applyRectSizingToHTMLElement(bodyPosition, fGet(bodyRef.current))
          applyPaddingToHTMLElement(cardPadding, fGet(bodyRef.current))

          const transitionReverse = linearFnFomPoints(0, 1, 1, 0)(transition)

          fGet(menuRef.current).setButtonScale(menuButtonScale)
          fGet(menuDivRef.current).style.top = `${menuPosition.y}px`
          fGet(menuDivRef.current).style.left = `${menuPosition.x}px`
          fGet(menuDivRef.current).style.width = `${menuPosition.width}px`
          fGet(descriptionRef.current).style.top = `${descriptionPosition.y}px`
          fGet(descriptionRef.current).style.left = `${descriptionPosition.x}px`
          fGet(
            descriptionRef.current
          ).style.width = `${descriptionPosition.width}px`
          fGet(descriptionRef.current).style.opacity = `${transitionReverse}`
          fGet(
            descriptionRef.current
          ).style.transform = `scale(${descriptionScale})`

          fGet(mainChartRef.current).setSizing(chartSizing)

          applyRectSizingToHTMLElement(
            legacyBodyPosition,
            fGet(legacyBodyRef.current)
          )
          fGet(
            legacyHeadingRef.current
          ).style.width = `${legacyBodyPosition.width}px`
          fGet(legacyHeadingRef.current).style.left = `0px`
          fGet(legacyHeadingRef.current).style.top = `${cardPadding.top}px`
          fGet(
            legacyHeadingRef.current
          ).style.transform = `scale(${menuButtonScale})`
          fGet(legacyChartRef.current).setSizing(legacySizing)

          applyPositionToHTMLElement(rescalePosition, fGet(rescaleRef.current))
          fGet(rescaleRef.current).style.height = `${rescalePosition.height}px`

          fGet(tasksRef.current).style.right = `${tasksPosition.right}px`
          fGet(tasksRef.current).style.top = `${tasksPosition.y}px`

          fGet(tasksRef.current).style.opacity = `${transitionReverse}`
          fGet(tasksRef.current).style.visibility =
            transition === 1 ? 'hidden' : 'visible'
        },
        [sizing]
      )
      useEffect(() => {
        const tween = gsap.to(legacyTransitionRef.current, {
          transition: shouldShowLegacy ? 1 : 0,
          onUpdate: () =>
            setTransition(
              transitionRef.current.transition,
              transitionRef.current.target,
              legacyTransitionRef.current.transition
            ),
        })
        return () => {
          tween.kill()
        }
      }, [shouldShowLegacy, setTransition, transitionRef])
      useAssertConst([transitionRef])

      useImperativeHandle(
        forwardRef,
        () => ({
          setTransition: (t, x) =>
            setTransition(t, x, legacyTransitionRef.current.transition),
        }),
        [setTransition]
      )
      useLayoutEffect(() => {
        setTransition(
          transitionRef.current.transition,
          transitionRef.current.target,
          legacyTransitionRef.current.transition
        )
      }, [setTransition, transitionRef])
      useAssertConst([transitionRef])

      return (
        <div
          className={`absolute top-0  right-0 z-10 bg-chartBG overflow-hidden
          ${layout == 'laptop' ? 'rounded-2xl' : ''}`}
          ref={outerRef}
        >
          {layout !== 'mobile' && (
            <h2
              className={`absolute uppercase font-bold text-white `}
              ref={headingRef}
            >
              Results
            </h2>
          )}
          <div
            className="absolute bg-cardBG   rounded-t-2xl rounded-b-2xl overflow-hidden"
            ref={bodyRef}
          >
            <div className="absolute" ref={menuDivRef}>
              <ChartPanelMenu
                type={state.type}
                onSelect={handleChangeType}
                layout={layout}
                ref={menuRef}
                showDescriptionPopUp={() => setShowDescriptionPopUp(true)}
              />
            </div>
            <div className="absolute invisible" ref={menuAt0DivRef}>
              <ChartPanelMenuButton
                type={state.type}
                onClick={() => {}}
                showDescriptionPopUp={() => {}}
                layout={layout}
                style={{
                  left: `${sizingAt0AtTargetLegacy.menuPosition.x}px`,
                  top: `${sizingAt0AtTargetLegacy.menuPosition.y}px`,
                  width: `${sizingAt0AtTargetLegacy.menuPosition.width}px`,
                  transform: `scale(${sizingAt0AtTargetLegacy.menuButtonScale})`,
                  transformOrigin: 'top left',
                }}
              />
            </div>
            <div className="absolute invisible" ref={menuAt1DivRef}>
              <ChartPanelMenuButton
                type={state.type}
                layout={layout}
                onClick={() => {}}
                showDescriptionPopUp={() => {}}
                style={{
                  left: `${sizingAt1AtTargetLegacy.menuPosition.x}px`,
                  top: `${sizingAt1AtTargetLegacy.menuPosition.y}px`,
                  width: `${sizingAt1AtTargetLegacy.menuPosition.width}px`,
                  transform: `scale(${sizingAt1AtTargetLegacy.menuButtonScale})`,
                  transformOrigin: 'top left',
                }}
              />
            </div>
            <div
              className="absolute"
              ref={descriptionRef}
              style={{transformOrigin: 'top left'}}
            >
              <ChartPanelDescription
                type={state.type}
                showDescriptionPopUp={showDescriptionPopUp}
                setShowDescriptionPopUp={setShowDescriptionPopUp}
              />
            </div>
            <div
              className="absolute invisible"
              ref={descriptionAt0Ref}
              style={{
                left: `${sizingAt0AtTargetLegacy.descriptionPosition.x}px`,
                top: `${sizingAt0AtTargetLegacy.descriptionPosition.y}px`,
                width: `${sizingAt0AtTargetLegacy.descriptionPosition.width}px`,
              }}
            >
              <ChartPanelDescription
                type={state.type}
                showDescriptionPopUp={false}
                setShowDescriptionPopUp={() => {}}
              />
            </div>
            <TPAWChartMain
              state={state.main}
              starting={{sizing: startingSizing.chartSizing}}
              ref={mainChartRef}
            />
          </div>
          <div
            className="absolute bg-cardBG rounded-2xl overflow-hidden"
            ref={legacyBodyRef}
          >
            <h2
              className="absolute font-bold text-lg sm:text-xl text-center"
              ref={legacyHeadingRef}
              style={{transformOrigin: 'top center'}}
            >
              Legacy
            </h2>
            <TPAWChartLegacy
              state={state.legacy}
              starting={{sizing: startingSizing.legacySizing}}
              ref={legacyChartRef}
            />
          </div>
          <button
            className={`absolute bg-cardBG px-3 py-1 flex gap-x-2 items-center rounded-full 
            ${rescaleWarningLevel === 2 ? ' text-red-500 ' : ''}`}
            style={{
              visibility: rescaleWarningLevel === 0 ? 'hidden' : 'visible',
              opacity: rescaleWarningLevel === 0 ? '0' : '1',
              transform: rescaleWarningLevel === 0 ? 'scale(.8)' : 'scale(1)',
              transitionProperty: 'transform, opacity, visibility',
              transitionDuration:'300ms'
            }}
            onClick={handleRescale}
            ref={rescaleRef}
          >
            <FontAwesomeIcon className="" icon={faArrowsUpDown} />
            Rescale
            {rescaleWarningLevel === 2 && (
              <>
                <div className="absolute -right-[2px] -top-[0px] w-[10px] h-[10px] bg-red-500 rounded-full"></div>
                <div className="absolute -right-[12px] -top-[10px] w-[30px] h-[30px] bg-red-500 rounded-full animate-ping "></div>
              </>
            )}
          </button>
          <Link href="/tasks-for-this-year">
            <a
              className="absolute whitespace-nowrap bg-cardB py-1 rounded-xl flex items-center gap-x-2 text-white"
              ref={tasksRef}
            >
              <h2 className="font-bold text-base sm:text-xl flex items-center gap-x-2">
                <FontAwesomeIcon icon={faClipboardList} />
                Tasks for this year
              </h2>
            </a>
          </Link>
        </div>
      )
    }
  )
)

const legacyBodyX = (
  {position, padding, legacyWidth}: ReturnType<Props['sizing']>,
  legacyTransition: number
) =>
  linearFnFomPoints(
    0,
    position.width + 2,
    1,
    position.width - legacyWidth - padding.right
  )(legacyTransition)

const bodyWFn = (
  sizing: ReturnType<Props['sizing']>,
  legacyTransition: number
) => {
  const {padding, intraGap} = sizing
  return (
    legacyBodyX(sizing, legacyTransition) -
    padding.left -
    linearFnFomPoints(0, padding.right, 1, intraGap)(legacyTransition)
  )
}

const bodyContentWFn = (
  sizing: ReturnType<Props['sizing']>,
  legacyTransition: number
) =>
  bodyWFn(sizing, legacyTransition) -
  sizing.cardPadding.left -
  sizing.cardPadding.right

const _rescaleWarningLevel = (
  curr: SimpleRange,
  target: SimpleRange
): 0 | 1 | 2 => {
  const length = Math.abs(target.end - target.start)
  const missing =
    (Math.max(0, curr.start - target.start) +
      Math.max(0, target.end - curr.end)) /
    length
  const excess =
    (Math.max(0, target.start - curr.start) +
      Math.max(0, curr.end - target.end)) /
    length

  if (missing > 0.5 || excess > 1.25) return 2
  if (missing > 0.025 || excess > 0.05) return 1
  return 0
}

const _maxRescaleWarningLevel = (w1: 0 | 1 | 2, w2: 0 | 1 | 2): 0 | 1 | 2 =>
  Math.max(w1, w2) as 0 | 1 | 2
