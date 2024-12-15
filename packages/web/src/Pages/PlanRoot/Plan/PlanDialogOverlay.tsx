import { block, fGet } from '@tpaw/common'
import React, { ReactNode, useEffect, useLayoutEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import { colorPalette, gray, orange } from '../../../Utils/ColorPalette'
import { RectExt, Size, newPadding, rectExt } from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { DialogBubble } from '../../Common/DialogBubble'
import { useSimulationInfo } from '../PlanRootHelpers/WithSimulation'
import { isPlanSectionDialogInOverlayMode } from './PlanInput/Helpers/PlanSectionDialogPosition'

export const PlanDialogOverlay = React.memo(
  ({
    chartDiv,
  }: {
    // Because this resizes with animation on window resizing.
    chartDiv: HTMLElement | null
  }) => {
    const { planParamsNormInstant, updatePlanParams, simulationInfoByMode } =
      useSimulationInfo()
    const { dialogPosition } = planParamsNormInstant
    if (
      !(
        isPlanSectionDialogInOverlayMode(dialogPosition.effective) ||
        dialogPosition.effective === 'done'
      ) ||
      // Don't show in history mode because it hides the calendar input, so
      // they will get stuck.
      simulationInfoByMode.mode === 'history'
    )
      return <></>
    const handleNext = () => {
      if (dialogPosition.effective === 'done') return
      updatePlanParams('setDialogPosition', dialogPosition.next)
    }

    return ReactDOM.createPortal(
      block(() => {
        switch (dialogPosition.effective) {
          case 'show-results':
            return (
              <_Overlay
                dialogPosition={dialogPosition.effective}
                chartDiv={chartDiv}
                elementId="planResultsDialogCurtionShowResultsButton"
                padding={{ horz: 0, vert: 0 }}
                // Need a fill to make clickable (at least on chrome.)
                fill={colorPalette.withOpacity(gray[50], 0.1)}
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
                      You have entered the minimum inputs needed to simulate
                      your retirement. Click here to view the results panel.
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
                fill={colorPalette.withOpacity(orange[200], 0.4)}
                // fill="none"
                border={null}
                rounding={() => 25}
                hidden={dialogPosition.effective === 'done'}
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
      }),
      window.document.body,
    )
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
