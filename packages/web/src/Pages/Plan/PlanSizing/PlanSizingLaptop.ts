import {rectExt} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {PlanSizing} from './PlanSizing'

export function planSizingLaptop(windowSize: {
  width: number
  height: number
}): PlanSizing {
  const pad = 30
  const topFn = (transition: number) =>
    linearFnFomPoints(0, 50, 1, 70)(transition)

  const headingMarginBottom = 20

  // ---- GUIDE ----
  const guide = (() => {
    const size = {
      width: windowSize.width * 0.28,
      height: windowSize.height,
    }
    const dynamic = (transition: number) => {
      return {
        origin: {
          x: linearFnFomPoints(0, -size.width + pad * 0.75, 1, 0)(transition),
          y: 0,
        },
      }
    }
    return {
      dynamic,
      fixed: {
        size,
        padding: {left: pad, right: pad * 0.75, top: topFn(1), bottom: pad},
        headingMarginBottom,
      },
    }
  })()

  // ---- INPUT SUMMARY ----
  const inputSummary = ((): PlanSizing['inputSummary'] => {
    const extraPad = 10
    return {
      dynamic: (transition: number) => ({
        origin: {
          x: guide.dynamic(transition).origin.x + guide.fixed.size.width,
          y: 0,
        },
      }),
      fixed: {
        size: {
          height: windowSize.height,
          width: Math.max(windowSize.width * 0.37, 350),
        },
        padding: {
          left: pad * 0.25 + extraPad,
          right: pad * 0.75 + extraPad,
          top: topFn(0),
          bottom: pad,
        },
        cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
      },
    }
  })()

  // ---- INPUT ----
  const input = ((): PlanSizing['input'] => {
    const dynamic = (transition: number) => ({
      origin: {
        x: inputSummary.dynamic(transition).origin.x,
        y: 0,
      },
    })
    return {
      dynamic,
      fixed: {
        size: {
          width: Math.max(windowSize.width * 0.3, 350),
          height: windowSize.height - dynamic(1).origin.y,
        },
        padding: {
          left: pad * 0.25,
          right: pad * 0.75,
          top: topFn(1),
          bottom: pad,
        },
        cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
        headingMarginBottom,
      },
    }
  })()

  // ---- HEADING -----
  const heading = (() => {
    const dynamic = (transition: number) => {
      return {
        origin: {
          x: guide.dynamic(transition).origin.x + pad,
          y: 0,
        },
      }
    }
    return {
      dynamic,
      fixed: {
        size: {
          width:
            input.dynamic(1).origin.x +
            input.fixed.size.width -
            dynamic(1).origin.x,
          height: 60,
        },
      },
    }
  })()

  // ---- CHART ----
  const chart = (transition: number) => {
    const vertInnerPadFn = linearFnFomPoints(0, 35, 1, 15)
    const vertInnerPad = vertInnerPadFn(transition)
    const hozrInnerPad = vertInnerPad * 1.25
    const positionFn = (transition: number) => {
      // const y = topFn(transition) - padTop
      const y = linearFnFomPoints(0, 0, 1, topFn(1) - vertInnerPadFn(1))(transition)
      const x =
        linearFnFomPoints(
          0,
          inputSummary.dynamic(0).origin.x + inputSummary.fixed.size.width,
          1,
          input.dynamic(1).origin.x + input.fixed.size.width
        )(transition) +
        pad * 0.25
      return {
        x,
        y,
        // width: windowSize.width - x - pad,
        width:
          windowSize.width - x - linearFnFomPoints(0, 0, 1, pad)(transition),
        // height: windowSize.height - 2 * y,
        height: windowSize.height - y,
      }
    }

    const position = positionFn(transition)
    const positionAt0 = positionFn(0)
    const positionAt1 = positionFn(1)

    const heightAt1 = Math.min(
      positionAt1.height,
      Math.max(400, positionAt1.width * 0.85)
    )
    position.height = linearFnFomPoints(
      0,
      positionAt0.height,
      1,
      heightAt1
    )(transition)

    return {
      position: rectExt(position),
      // padding: {left: 20, right: 20, top: 10, bottom: 10},
      padding: {
        left: hozrInnerPad,
        right: hozrInnerPad,
        top: vertInnerPad,
        bottom: vertInnerPad,
      },
      // 18px is text-lg 20px is text-xl.
      menuButtonScale: linearFnFomPoints(0, 1, 22/22, 18 / 22)(transition),
      cardPadding: {left: 15, right: 15, top: 10, bottom: 10},
      headingMarginBottom: 10,
      legacyWidth: linearFnFomPoints(0, 120, 1, 100)(transition),
      intraGap: linearFnFomPoints(0, 20, 1, 10)(transition),
      borderRadius: linearFnFomPoints(0, 0, 1, 16)(transition),
    }
  }

  return {input, inputSummary, guide, chart, heading}
}
