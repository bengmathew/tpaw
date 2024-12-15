import { DialogPosition, fGet, noCase } from '@tpaw/common'
import React, { useEffect, useLayoutEffect, useState } from 'react'
import { RectExt, rectExt } from '../../../../Utils/Geometry'
import { DialogBubble } from '../../../Common/DialogBubble'
import { useSimulationInfo } from '../../PlanRootHelpers/WithSimulation'

export const PlanSummaryDialog = React.memo(
  ({
    dialogPosition,
    elements,
    fixedSizing,
  }: {
    elements: {
      outer: HTMLElement
      body: HTMLElement
      age: HTMLElement
      currentPortfolioBalance: HTMLElement
      futureSavings: HTMLElement
      incomeDuringRetirement: HTMLElement
      adjustmentsToSpending: HTMLElement
    }
    dialogPosition: Exclude<DialogPosition, 'done'>
    fixedSizing: { padding: { top: number } }
  }) => {
    const { updatePlanParams, planParamsNormInstant } = useSimulationInfo()

    const [measures, setMeasures] = useState({
      age: rectExt(0),
      'current-portfolio-balance': rectExt(0),
      'future-savings': rectExt(0) as RectExt | null,
      'income-during-retirement': rectExt(0),
      'adjustments-to-spending': rectExt(0),
    })

    useLayoutEffect(() => {
      const outer = elements.outer
      const scroll = (
        target: keyof typeof elements,
        opts: {
          behavior?: 'smooth' | 'auto'
          delay?: boolean
          modify?: (
            x: RectExt,
            currVisible: RectExt,
            outerRect: RectExt,
          ) => RectExt
        } = {},
      ) => {
        const doFn = () => {
          const modify =
            opts.modify ??
            ((x: RectExt) =>
              rectExt({
                y: x.y - 10,
                bottom: x.bottom + 100,
                x: 0,
                width: 0,
              }))

          const behavior = opts.behavior ?? 'smooth'

          const outerRect = rectExt(outer.getBoundingClientRect())
          const currVisible = rectExt({
            y: outer.scrollTop,
            height: outerRect.height,
            x: 0,
            width: 0,
          })

          const _rel = _relativePosition(
            fGet(elements.body).getBoundingClientRect(),
            fixedSizing.padding.top,
          )

          const measure = modify(
            rectExt(_rel(fGet(elements[target]))),
            currVisible,
            rectExt(outerRect),
          )

          if (measure.y < currVisible.y) {
            outer.scrollTo({ top: measure.y, behavior })
          } else if (measure.bottom > currVisible.bottom) {
            const top = measure.bottom - outerRect.height
            outer.scrollTo({ top, behavior })
          }
        }

        if (opts.delay) {
          window.setTimeout(doFn, 100)
        } else {
          doFn()
        }
      }
      switch (dialogPosition) {
        case 'age':
          outer.scrollTo({ top: 0, behavior: 'smooth' })
          break
        case 'current-portfolio-balance':
          scroll('currentPortfolioBalance')
          break
        case 'future-savings':
          scroll('futureSavings')
          break
        case 'income-during-retirement':
          scroll('incomeDuringRetirement')
          break
        case 'show-results':
        case 'show-all-inputs': {
          // scrolling on  show-all-inputs is needed for loading directly into 'show-all-inputs'.
          scroll('adjustmentsToSpending', {
            // Delay is because on desktop the outer height may change at start of
            // show-results and we need to get the height after the change.
            delay: true,
            behavior: 'auto',
            modify: (measure) =>
              rectExt({
                y: measure.y - 50,
                bottom: measure.y + 150,
                x: 0,
                width: 0,
              }),
          })
          break
        }
        default:
          noCase(dialogPosition)
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dialogPosition])

    useLayoutEffect(() => {
      if (
        !elements.body ||
        !elements.age ||
        !elements.currentPortfolioBalance ||
        !elements.incomeDuringRetirement ||
        !elements.adjustmentsToSpending
      )
        return

      const bodyElement = elements.body
      const resizeObserver = new ResizeObserver(() => {
        const bodyRect = bodyElement.getBoundingClientRect()
        const _rel = _relativePosition(bodyRect, fixedSizing.padding.top)
        setMeasures({
          age: _rel(elements.age),
          'current-portfolio-balance': _rel(elements.currentPortfolioBalance),
          'future-savings': _rel(elements.futureSavings, true),
          'income-during-retirement': _rel(elements.incomeDuringRetirement),
          ['adjustments-to-spending']: _rel(elements.adjustmentsToSpending),
        })
      })
      resizeObserver.observe(bodyElement)
      return () => resizeObserver.disconnect()
    }, [
      elements.adjustmentsToSpending,
      elements.age,
      elements.body,
      elements.currentPortfolioBalance,
      elements.futureSavings,
      elements.incomeDuringRetirement,
      fixedSizing.padding.top,
    ])

    return (
      <>
        {dialogPosition === 'show-all-inputs' ||
        dialogPosition === 'show-results' ? (
          <></>
        ) : (
          <DialogBubble
            arrowOffset={40}
            arrowDirection="up"
            x={fGet(measures[dialogPosition]).right - 300}
            y={fGet(measures[dialogPosition]).bottom}
            zIndex={10}
          >
            {dialogPosition === 'age' ? (
              'Start Here'
            ) : (
              <div className="">
                <h2 className="">Continue Here</h2>
                <div className="mt-1 mb-1">
                  <span className="mr-1">or </span>
                  <button
                    className="btn-sm bg-orange-200  rounded-full text-base"
                    onClick={() =>
                      updatePlanParams(
                        'setDialogPosition',
                        planParamsNormInstant.dialogPosition.next,
                      )
                    }
                  >
                    Skip This Section
                  </button>
                </div>
              </div>
            )}
          </DialogBubble>
        )}
      </>
    )
  },
)

const _relativePosition =
  (bodyRect: DOMRect, paddingTop: number) =>
  (element: HTMLElement | null, trace = false) => {
    const x = fGet(element).getBoundingClientRect()
    return rectExt({
      x: x.left,
      width: x.width,
      // fixedSizing.padding.top to account for the marginTop of the
      // sticky Save/Reset div which is causing a margin on the bodyDiv
      // which I don't fully understand.
      y: x.top - bodyRect.y + paddingTop,
      height: x.height,
    })
  }
