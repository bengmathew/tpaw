import {rectExt} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {PlanSizing} from './PlanSizing'

export function planSizingDesktop(windowSize: {
  width: number
  height: number
}): PlanSizing {
  const pad = 20

  const navHeadingH = 80
  const headingMarginBottom = 10

  const chart = (transition: number) => {
    const inset = 0
    let height =
      windowSize.width < 800
        ? linearFnFomPoints(400, 300, 800, 550)(windowSize.width)
        : 550

    const padBottom = (transition: number) =>
      linearFnFomPoints(0, pad, 1, pad / 2)(transition)
    height -= linearFnFomPoints(0, 0, 1, navHeadingH / 2)(transition)
    return {
      position: rectExt({
        width: windowSize.width - inset * 2,
        height: height,
        x: inset,
        y: inset,
      }),
      padding: {
        left: pad * 2,
        right: pad * 2,
        top: pad,
        bottom: padBottom(transition),
      },
      cardPadding: {left: 15, right: 15, top: 10, bottom: 10},
      headingMarginBottom,
      menuButtonScale: 1,
      legacyWidth: 120,
      intraGap: pad,
      borderRadius:0
    }
  }

  // ---- HEADING ----
  const heading = (() => {
    const dynamic = (transition: number) => {
      return {
        origin: {
          x: pad * 2 + linearFnFomPoints(0, 0, 1, 0)(transition),
          y: chart(transition).position.bottom,
        },
      }
    }
    return {
      dynamic,
      fixed: {size: {width: windowSize.width - pad * 2, height: navHeadingH}},
    }
  })()

  // ---- INPUT SUMMARY ----
  const inputSummary = ((): PlanSizing['inputSummary'] => {
    const dynamic = (transition: number) => ({
      origin: {x: 0, y: chart(transition).position.bottom},
    })
    return {
      dynamic,
      fixed: {
        size: {
          width: windowSize.width,
          height: windowSize.height - dynamic(1).origin.y,
        },
        padding: {left: pad * 2, right: pad * 1.75, top: 20, bottom: 0},
        cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
      },
    }
  })()

  // ---- INPUT ----
  const input = (() => {
    const dynamic = (transition: number) => ({
      origin: {
        x: 0,
        y: chart(transition).position.bottom,
      },
    })
    return {
      dynamic,
      fixed: {
        size: {
          width: windowSize.width * 0.5,
          height: windowSize.height - dynamic(1).origin.y,
        },
        padding: {
          left: pad * 2,
          right: pad * 1.75,
          top: heading.fixed.size.height,
          bottom: pad,
        },
        cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
        headingMarginBottom,
      },
    }
  })()

  // ---- GUIDE ----
  const guide = (() => {
    const dynamic = (transition: number) => {
      const currInputOrigin = input.dynamic(transition).origin
      return {
        origin: {
          x: currInputOrigin.x + input.fixed.size.width,
          y: currInputOrigin.y,
        },
      }
    }
    return {
      dynamic,
      fixed: {
        size: {
          width: windowSize.width - dynamic(1).origin.x,
          height: input.fixed.size.height,
        },
        padding: {
          left: pad * 0.25,
          right: pad * 2,
          top: input.fixed.padding.top,
          bottom: input.fixed.padding.bottom,
        },
        cardPadding: {left: 0, right: 0, top: 0, bottom: 0},
        headingMarginBottom,
      },
    }
  })()

  return {input, inputSummary, guide, chart, heading}
}
