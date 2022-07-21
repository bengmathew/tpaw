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
      }
    }

    const heading = (transition: number) => {
      return {
        position: rectExt({
          width: windowSize.width - pad * 2,
          height: navHeadingH,
          x: pad * 2,
          y: chart(transition).position.bottom + 10,
        }),
      }
    }

    // ---- INPUT SUMMARY ----
    const inputSummary = ((): PlanSizing['inputSummary'] => ({
      dynamic: (transition: number) => {
        const y = chart(transition).position.bottom
        return {
          region: rectExt({
            width: windowSize.width,
            height: windowSize.height - y,
            x: 0,
            y,
          }),
        }
      },
      fixed: {
        padding: {left: pad, right: pad, top: 20, bottom: 0},
        cardPadding: {left: 10, right: 10, top: 10, bottom: 10},
      },
    }))()

    const input = (() => ({
      dynamic: (transition: number) => {
        return {
          region: rectExt({
            x: 0,
            y: chart(transition).position.bottom + heading(1).position.height,
            width: windowSize.width,
            bottom: windowSize.height,
          }),
        }
      },
      fixed: {
        padding: {
          left: pad,
          right: pad,
          top: navHeadingMarginBottom,
          bottom: 0,
        },
        cardPadding: {left: 10, right: 10, top: 10, bottom: 10},
        headingMarginBottom: 10,
      },
    }))()

    const headingMarginBottom = 10

    const guide = (transition: number) => {
      const pad = 10
      const height = 50
      const width = 100
      return {
        position: rectExt({
          width,
          height,
          x: windowSize.width - width - pad,
          y:
            windowSize.height -
            height -
            pad +
            linearFnFomPoints(0, height + pad, 1, 0)(transition),
        }),
        padding: {left: pad, right: pad, top: pad, bottom: pad},
        cardPadding: {left: 0, right: 0, top: 0, bottom: 0},
        headingMarginBottom: 0,
      }
    }
    return {input, inputSummary, guide, chart, heading}
  }
}
