import { fGet, PlanParams } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useLayoutEffect, useState } from 'react'
import { RectExt, rectExt } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { DialogBubble } from '../../Common/DialogBubble'
import { nextPlanSectionDialogPosition } from '../PlanInput/Helpers/PlanSectionDialogPosition'

export const PlanSummaryDialog = React.memo(
  ({
    dialogPosition,
    elements,
    fixedSizing,
  }: {
    elements: {
      outer: HTMLElement | null
      body: HTMLElement | null
      age: HTMLElement | null
      currentPortfolioBalance: HTMLElement | null
      futureSavings: HTMLElement | null
      incomeDuringRetirement: HTMLElement | null
      adjustmentsToSpending: HTMLElement | null
    }
    dialogPosition: Exclude<PlanParams['dialogPosition'], 'done'>
    fixedSizing: { padding: { top: number } }
  }) => {
    const { setParams, paramsExt } = useSimulation()
    const { withdrawalStartMonth, asMFN } = paramsExt

    const [measures, setMeasures] = useState({
      age: rectExt(0),
      'current-portfolio-balance': rectExt(0),
      'future-savings': rectExt(0) as RectExt | null,
      'income-during-retirement': rectExt(0),
      'adjustments-to-spending': rectExt(0),
    })

    const [forceScroll, setForceScroll] = useState(0)

    useLayoutEffect(() => {
      const outer = elements.outer
      if (!outer) return
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
        case 'show-results': {
          scroll('adjustmentsToSpending', {
            // Delay is because on desktop the outer height may change at start of
            // show-results and we need to get the height after the change.
            delay: true,
            behavior: 'auto',
            modify: (measure) => {
              return rectExt({
                y: measure.y - 50,
                bottom: measure.y + 150,
                x: 0,
                width: 0,
              })
            },
          })
          break
        }
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dialogPosition, forceScroll])

    useEffect(() => {
      // Initial scroll. Delay is needed to get measures.
      window.setTimeout(() => setForceScroll((x) => x + 1), 100)
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useLayoutEffect(() => {
      if (
        !elements.body ||
        !elements.age ||
        !elements.currentPortfolioBalance ||
        !elements.incomeDuringRetirement ||
        !elements.adjustmentsToSpending
      )
        return

      const resizeObserver = new ResizeObserver(() => {
        const bodyRect = fGet(elements.body).getBoundingClientRect()
        const _rel = _relativePosition(bodyRect, fixedSizing.padding.top)
        setMeasures({
          age: _rel(elements.age),
          'current-portfolio-balance': _rel(elements.currentPortfolioBalance),
          'future-savings': elements.futureSavings
            ? _rel(elements.futureSavings, true)
            : null,
          'income-during-retirement': _rel(elements.incomeDuringRetirement),
          ['adjustments-to-spending']: _rel(elements.adjustmentsToSpending),
        })
      })
      resizeObserver.observe(elements.body)
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
                    onClick={() => {
                      setParams((params) => {
                        const clone = _.cloneDeep(params)
                        clone.dialogPosition = nextPlanSectionDialogPosition(
                          dialogPosition,
                          asMFN(withdrawalStartMonth),
                        )
                        return clone
                      })
                    }}
                  >
                    Skip This Section
                  </button>
                </div>
              </div>
            )}
          </DialogBubble>
        )}

        {/* {dialogPosition !== 'show-results' &&
          dialogPosition !== 'show-all-inputs' && (
            <div
              className="absolute bg-planBG bg-opacity-80 z-10"
              style={{
                top: `${fGet(measures[dialogPosition]).bottom}px`,
                bottom: '0px',
                width: '100%',
                left: '0px',
              }}
            />
          )} */}
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
