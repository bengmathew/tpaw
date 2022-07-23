import {rectExt} from '../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../Utils/LinearFn'
import {headerHeight} from '../../App/Header'
import {PlanSizing} from './PlanSizing'

export function planSizingMobile(windowSize: {
  width: number
  height: number
}): PlanSizing {
  {
    const pad = 10

    const navHeadingH = 40
    const navHeadingMarginBottom = 20
    const headingMarginBottom = 10

    const chart = (transition: number) => {
      let height = linearFnFomPoints(375, 330, 415, 355)(windowSize.width)

      height -= linearFnFomPoints(
        0,
        0,
        1,
        (navHeadingH + navHeadingMarginBottom) * 0.25
      )(transition)
      return {
        position: rectExt({width: windowSize.width, height, x: 0, y: 0}),
        padding: {
          left: pad,
          right: pad,
          top: headerHeight + 5,
          bottom: pad,
        },
        cardPadding: {left: 15, right: 15, top: 10, bottom: 10},
        headingMarginBottom,
        menuButtonScale: 1,
        legacyWidth: 100,
        intraGap: pad,
        borderRadius:0
      }
    }

    // ---- HEADING ----
    const heading = ((): PlanSizing['heading'] => ({
      dynamic: (transition: number) => ({
        origin: {
          x: pad * 2,
          y: chart(transition).position.bottom + 10,
        },
      }),
      fixed: {
        size: {
          width: windowSize.width - pad * 2,
          height: navHeadingH,
        },
      },
    }))()

    // ---- INPUT SUMMARY ----
    const inputSummary = ((): PlanSizing['inputSummary'] => {
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
            width: windowSize.width,
            height: windowSize.height - dynamic(1).origin.y,
          },
          padding: {left: pad, right: pad, top: 20, bottom: 0},
          cardPadding: {left: 10, right: 10, top: 10, bottom: 10},
        },
      }
    })()

    // ---- INPUT SUMMARY ----
    const input = (() => {
      const dynamic = (transition: number) => {
        return {
          origin: {
            x: 0,
            y: chart(transition).position.bottom,
          },
        }
      }
      return {
        dynamic,
        fixed: {
          size: {
            width: windowSize.width,
            height: windowSize.height - dynamic(1).origin.y,
          },
          padding: {
            left: pad,
            right: pad,
            top: heading.fixed.size.height + navHeadingMarginBottom,
            bottom: 0,
          },
          cardPadding: {left: 10, right: 10, top: 10, bottom: 10},
          headingMarginBottom: 10,
        },
      }
    })()

    // ---- GUIDE ----
    const guide = (() => {
      const size = {
        width: 100,
        height: 50,
      }
      const dynamic = (transition: number) => {
        const pad = 10
        return {
          origin: {
            x: windowSize.width - size.width - pad,
            y:
              windowSize.height -
              size.height -
              pad +
              linearFnFomPoints(0, size.height + pad, 1, 0)(transition),
          },
        }
      }
      return {
        dynamic,
        fixed: {
          size,
          padding: {left: pad, right: pad, top: pad, bottom: pad},
          cardPadding: {left: 0, right: 0, top: 0, bottom: 0},
          headingMarginBottom: 0,
        },
      }
    })()
    return {input, inputSummary, guide, chart, heading}
  }
}
