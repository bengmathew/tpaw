import {faArrowsUpDown} from '@fortawesome/pro-light-svg-icons'
import {faClipboardList} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import gsap, {Power1, Power4} from 'gsap'
import Link from 'next/link'
import React, {
  Dispatch,
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
import {useChartData} from '../../App/WithChartData'
import {SimulationInfo, useSimulation} from '../../App/WithSimulation'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {ChartReactStatefull} from '../../Common/Chart/ChartReact'
import {Config} from '../../Config'
import {ChartPanelDescription} from './ChartPanelDescription'
import {ChartPanelMenu, ChartPanelMenuStateful} from './ChartPanelMenu'
import {ChartPanelMenuButton} from './ChartPanelMenuButton'
import {ChartPanelType} from './ChartPanelType'
import {TPAWChartDataLegacy} from './TPAWChart/TPAWChartDataLegacy'
import {
  TPAWChartDataMain,
  tpawChartDataScaled,
} from './TPAWChart/TPAWChartDataMain'
import {TPAWChartLegacy} from './TPAWChart/TPAWChartLegacy'
import {TPAWChartMain} from './TPAWChart/TPAWChartMain'

const morphAnimation: ChartAnimation = {ease: Power4.easeOut, duration: 1.5}
const normalAnimation: ChartAnimation = {ease: Power1.easeOut, duration: 0.5}

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
  type: ChartPanelType | 'sharpe-ratio'
  setType: Dispatch<ChartPanelType | 'sharpe-ratio'>
}

export const ChartPanel = React.memo(
  React.forwardRef<ChartPanelStateful, Props>(
    (
      {type, setType, layout, sizing: sizingIn, transitionRef}: Props,
      forwardRef
    ) => {
      const {paramSpace, setParamSpace, tpawResult} = useSimulation()
      const shouldShowLegacy = _shouldShowLegacy(tpawResult, type)
      const shouldShowSuccessRate = tpawResult.args.params.strategy === 'SWR' && type !== 'sharpe-ratio'
      const allChartData = useChartData()
      const chartMainData =
        type === 'sharpe-ratio'
          ? fGet(allChartData.sharpeRatio)
          : fGet(allChartData.byYearsFromNowPercentiles.get(type))
      const chartLegacyData = allChartData.legacy

      const mainChartRef =
        useRef<ChartReactStatefull<TPAWChartDataMain> | null>(null)

      const [yRange, setYRange] = useState({
        main: chartMainData.yDisplayRange,
        legacy: chartLegacyData.xyDisplayRange.y,
      })
      useEffect(() => {
        const mainChart = mainChartRef.current
        if (!mainChart) return
        const data = chartMainData
        const prevState = mainChart.getState()
        if (type === prevState.data.type) {
          mainChart.setState(
            data,
            {x: data.years.displayRange, y: prevState.xyRange.y},
            normalAnimation
          )
        } else {
          mainChart.setState(
            tpawChartDataScaled(prevState.data, data.yDisplayRange),
            {x: prevState.xyRange.x, y: data.yDisplayRange},
            null
          )
          mainChart.setState(
            data,
            {x: data.years.displayRange, y: data.yDisplayRange},
            morphAnimation
          )
        }
        setYRange(({legacy}) => ({
          main: mainChart.getState().xyRange.y,
          legacy,
        }))
      }, [chartMainData, type])

      useEffect(() => {
        const legacyChart = legacyChartRef.current
        if (!legacyChart) return
        const data = chartLegacyData
        const prevState = legacyChart.getState()
        legacyChart.setState(data, prevState.xyRange, normalAnimation)
      }, [chartLegacyData])

      const handleRescale = () => {
        const mainChart = mainChartRef.current
        const legacyChart = legacyChartRef.current
        if (!mainChart || !legacyChart) return
        mainChart.setState(
          chartMainData,
          {x: chartMainData.years.displayRange, y: chartMainData.yDisplayRange},
          normalAnimation
        )
        legacyChart.setState(
          chartLegacyData,
          chartLegacyData.xyDisplayRange,
          normalAnimation
        )
        setYRange({
          main: mainChart.getState().xyRange.y,
          legacy: legacyChart.getState().xyRange.y,
        })
      }

      const rescaleWarningLevel = _maxRescaleWarningLevel(
        _rescaleWarningLevel(yRange.main, chartMainData.yDisplayRange),
        shouldShowLegacy
          ? _rescaleWarningLevel(
              yRange.legacy,
              chartLegacyData.xyDisplayRange.y
            )
          : 0
      )
      const [pingRescale, setPingRescale] = useState(false)
      useEffect(() => setPingRescale(true), [rescaleWarningLevel])
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
      const legacyBodyRef = useRef<HTMLDivElement | null>(null)
      const legacyHeadingRef = useRef<HTMLHeadingElement | null>(null)
      const legacyChartRef =
        useRef<ChartReactStatefull<TPAWChartDataLegacy> | null>(null)
      const tasksRef = useRef<HTMLAnchorElement | null>(null)
      const rescaleRef = useRef<HTMLButtonElement | null>(null)
      const legacyTransitionRef = useRef({transition: shouldShowLegacy ? 1 : 0})
      const successRateTransitionRef = useRef({
        transition: shouldShowSuccessRate ? 1 : 0,
      })

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
        (
          transition: number,
          target: 0 | 1,
          legacyTransition: number,
          successRateTransition: number
        ) => {
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
            bottom: linearFnFomPoints(0, 35, 1, 60)(successRateTransition),
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
          legacyTransitionRef.current.transition,
          successRateTransitionRef.current.transition
        )
      )
      const sizingAt01 = useMemo(() => sizing(0, 0, 1, 0), [sizing])
      const sizingAt00 = useMemo(() => sizing(0, 0, 0, 0), [sizing])
      const sizingAt11 = useMemo(() => sizing(1, 1, 1, 0), [sizing])
      const sizingAt10 = useMemo(() => sizing(1, 1, 0, 0), [sizing])
      const sizingAt0AtTargetLegacy = shouldShowLegacy ? sizingAt01 : sizingAt00
      const sizingAt1AtTargetLegacy = shouldShowLegacy ? sizingAt11 : sizingAt10

      const setTransition = useCallback(
        (
          transition: number,
          target: 0 | 1,
          legacyTransition: number,
          successRateTransition: number
        ) => {
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
          } = sizing(
            transition,
            target,
            legacyTransition,
            successRateTransition
          )

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
          mainChartRef.current?.setSizing(chartSizing)

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
          legacyChartRef.current?.setSizing(legacySizing)

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
              legacyTransitionRef.current.transition,
              successRateTransitionRef.current.transition
            ),
        })
        return () => {
          tween.kill()
        }
      }, [shouldShowLegacy, setTransition, transitionRef])
      useAssertConst([transitionRef])
      useEffect(() => {
        const tween = gsap.to(successRateTransitionRef.current, {
          transition: shouldShowSuccessRate ? 1 : 0,
          onUpdate: () =>
            setTransition(
              transitionRef.current.transition,
              transitionRef.current.target,
              legacyTransitionRef.current.transition,
              successRateTransitionRef.current.transition
            ),
        })
        return () => {
          tween.kill()
        }
      }, [shouldShowSuccessRate, setTransition, transitionRef])
      useAssertConst([transitionRef])

      useImperativeHandle(
        forwardRef,
        () => ({
          setTransition: (t, x) =>
            setTransition(
              t,
              x,
              legacyTransitionRef.current.transition,
              successRateTransitionRef.current.transition
            ),
        }),
        [setTransition]
      )
      useLayoutEffect(() => {
        setTransition(
          transitionRef.current.transition,
          transitionRef.current.target,
          legacyTransitionRef.current.transition,
          successRateTransitionRef.current.transition
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
              onClick={
                Config.client.production
                  ? undefined
                  : () => setParamSpace(paramSpace === 'a' ? 'b' : 'a')
              }
            >
              Results {paramSpace === 'a' ? '' : '(B)'}
            </h2>
          )}
          <div
            className="absolute bg-cardBG   rounded-t-2xl rounded-b-2xl overflow-hidden"
            ref={bodyRef}
          >
            <div className="absolute" ref={menuDivRef}>
              <ChartPanelMenu
                type={type}
                onSelect={setType}
                layout={layout}
                ref={menuRef}
                showDescriptionPopUp={() => setShowDescriptionPopUp(true)}
              />
            </div>
            <div className="absolute invisible" ref={menuAt0DivRef}>
              <ChartPanelMenuButton
                type={type}
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
                type={type}
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
                type={type}
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
                type={type}
                showDescriptionPopUp={false}
                setShowDescriptionPopUp={() => {}}
              />
            </div>
            <TPAWChartMain
              starting={{
                data: chartMainData,
                xyRange: {
                  x: chartMainData.years.displayRange,
                  y: chartMainData.yDisplayRange,
                },
                sizing: startingSizing.chartSizing,
              }}
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
              starting={{
                data: chartLegacyData,
                xyRange: chartLegacyData.xyDisplayRange,
                sizing: startingSizing.legacySizing,
              }}
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
              transitionDuration: '300ms',
            }}
            onClick={handleRescale}
            ref={rescaleRef}
          >
            <FontAwesomeIcon className="" icon={faArrowsUpDown} />
            Rescale
            {rescaleWarningLevel === 2 && (
              <>
                <div className="absolute -right-[2px] -top-[0px] w-[10px] h-[10px] bg-red-500 rounded-full"></div>
                {pingRescale && (
                  <div
                    className="absolute -right-[12px] -top-[10px] w-[30px] h-[30px] bg-red-500 rounded-full  "
                    onAnimationEnd={() => setPingRescale(false)}
                    style={{animation: 'ping 1s cubic-bezier(0, 0, 0.2, 1) 4'}}
                  ></div>
                )}
                {/* Forces tailwind to insert ping keyframes. */}
                <div className="hidden animate-ping" />
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

const _shouldShowLegacy = (
  {args}: SimulationInfo['tpawResult'],
  type: ChartPanelType | 'sharpe-ratio'
) =>
  type === 'sharpe-ratio'
    ? false
    : args.params.strategy === 'SWR'
    ? true
    : args.params.legacy.total > 0 || args.params.spendingCeiling !== null
