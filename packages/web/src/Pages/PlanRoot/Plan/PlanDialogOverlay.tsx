import { fGet } from '@tpaw/common'
import React, { ReactNode, useEffect, useLayoutEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import { RectExt, Size, newPadding, rectExt } from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { ChartUtils } from '../../Common/Chart/ChartUtils/ChartUtils'
import { DialogBubble } from '../../Common/DialogBubble'
import {
  isPlanSectionDialogInOverlayMode,
  nextPlanSectionDialogPosition,
} from './PlanInput/Helpers/PlanSectionDialogPosition'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'
import { colors, gray, orange } from './UsePlanColors'

export const PlanDialogOverlay = React.memo(
  ({
    chartDiv,
  }: {
    // Because this resizes with animation on window resizing.
    chartDiv: HTMLElement | null
  }) => {
    const { planParams, updatePlanParams, simulationInfoByMode } =
      useSimulation()
    const { dialogPosition } = planParams
    if (
      !(
        isPlanSectionDialogInOverlayMode(dialogPosition) ||
        dialogPosition === 'done'
      ) ||
      // Don't show in history mode because it hides the calendar input, so
      // they will get stuck.
      simulationInfoByMode.mode === 'history'
    )
      return <></>
    const handleNext = () => {
      if (dialogPosition === 'done') return
      updatePlanParams(
        'setDialogPosition',
        nextPlanSectionDialogPosition(dialogPosition),
      )
    }
    const body = (() => {
      switch (dialogPosition) {
        case 'show-results':
          return (
            <_Overlay
              dialogPosition={dialogPosition}
              chartDiv={chartDiv}
              elementId="planResultsDialogCurtionShowResultsButton"
              padding={{ horz: 0, vert: 0 }}
              // Need a fill to make clickable (at least on chrome.)
              fill={colors.withOpacity(gray[50], 0.1)}
              border={null}
              rounding={(size) => size.height / 2}
              hidden={false}
              onClick={handleNext}
            >
              {(targetSize) => (
                <DialogBubble
                  arrowOffset={100}
                  arrowDirection={'up'}
                  x={targetSize.x + 100}
                  y={targetSize.bottom}
                  zIndex={0}
                >
                  <div className="max-w-[260px] sm:max-w-[320px] text-base  ">
                    You have entered the minimum inputs needed to simulate your
                    retirement. Click here to view the results panel.
                  </div>
                </DialogBubble>
              )}
            </_Overlay>
          )
        case 'show-all-inputs':
        case 'done':
          return (
            <_Overlay
              dialogPosition={'show-all-inputs'}
              chartDiv={chartDiv}
              elementId="planSummaryInputsAfterDialog"
              padding={{ horz: 20, vert: 20 }}
              fill={colors.withOpacity(orange[200], 0.4)}
              // fill="none"
              border={null}
              rounding={() => 25}
              hidden={dialogPosition === 'done'}
              onClick={null}
            >
              {(targetSize) => (
                <DialogBubble
                  arrowOffset={40}
                  arrowDirection={'down'}
                  x={targetSize.x + 80}
                  y={targetSize.y}
                  zIndex={0}
                >
                  <div className="max-w-[260px] sm:max-w-[320px] text-base ">
                    There are additional inputs you can enter here.
                  </div>
                  {
                    <div className="flex justify-end mt-2">
                      <button
                        className="btn-sm rounded-full bg-orange-200"
                        onClick={handleNext}
                      >
                        OK
                      </button>
                    </div>
                  }
                </DialogBubble>
              )}
            </_Overlay>
          )
        default:
          return <></>
      }
    })()

    return ReactDOM.createPortal(body, window.document.body)
  },
)

const _Overlay = React.memo(
  ({
    chartDiv,
    dialogPosition,
    children,
    padding,
    elementId,
    onClick,
    rounding,
    fill,
    border,
    hidden: hiddenIn,
  }: {
    dialogPosition: 'show-results' | 'show-all-inputs'
    // Because this resizes with animation on window resizing.
    chartDiv: HTMLElement | null
    padding: { horz: number; vert: number }
    elementId: string
    onClick: (() => void) | null
    border: { size: number; color: string } | null
    fill: string | null
    rounding: (target: RectExt) => number
    hidden: boolean
    children: (target: RectExt) => ReactNode
  }) => {
    const [outerDiv, setBodyElement] = useState<HTMLElement | null>(null)
    const [size, setSize] = useState<(Size & { target: RectExt }) | null>(null)

    const [hidden, setHidden] = useState(true)
    useEffect(() => setHidden(hiddenIn), [hiddenIn])

    useLayoutEffect(() => {
      if (!chartDiv || !outerDiv) return
      const rs = new ResizeObserver(() => {
        const { width, height } = outerDiv.getBoundingClientRect()
        const target = rectExt.inset(
          rectExt(
            fGet(document.getElementById(elementId)).getBoundingClientRect(),
          ),
          newPadding({ horz: -padding.horz, vert: -padding.vert }),
        )
        setSize({ width, height, target })
      })
      rs.observe(chartDiv, { box: 'border-box' })
      return () => rs.disconnect()
      // Resize observes chartOuterDiv not outerDiv because the chartOuterDiv
      // resizing captures everything, the resize on window size change and the
      // subsequent animation.
      //
    }, [
      chartDiv,
      dialogPosition,
      outerDiv,
      elementId,
      padding.horz,
      padding.vert,
    ])

    return (
      <NoDisplayOnOpacity0Transition
        ref={setBodyElement}
        className="absolute inset-0 page overflow-visible"
        style={{
          transitionProperty: 'opacity',
          transitionDuration: '1000ms',
          opacity: hidden ? '0' : '1',
        }}
      >
        {size && (
          <>
            <svg
              width={size.width}
              height={size.height}
              viewBox={`0 0 ${size.width} ${size.height}`}
            >
              <defs>
                <rect
                  id="target"
                  x={size.target.x}
                  y={size.target.y}
                  width={size.target.width}
                  height={size.target.height}
                  rx={rounding(size.target)}
                />
              </defs>

              <mask id="planChartHelpMask">
                <rect
                  x="0"
                  y="0"
                  width={size.width}
                  height={size.height}
                  fill="white"
                />
                <use href="#target" fill="black" />
              </mask>

              <rect
                className="opacity-80"
                width={size.width}
                height={size.height}
                x={0}
                y={0}
                fill={gray[300]}
                mask="url(#planChartHelpMask)"
              />
              <use
                href="#target"
                className={`${onClick ? 'cursor-pointer' : ''}`}
                strokeWidth={border?.size ?? 0}
                stroke={border?.color ?? 'none'}
                fill={fill ?? 'none'}
                onClick={onClick ?? undefined}
              />
            </svg>
            {children(size.target)}
          </>
        )}
      </NoDisplayOnOpacity0Transition>
    )
  },
)
