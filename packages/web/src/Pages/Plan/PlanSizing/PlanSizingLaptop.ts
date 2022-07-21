import { rectExt } from '../../../Utils/Geometry'
import { linearFnFomPoints } from '../../../Utils/LinearFn'
import { PlanSizing } from './PlanSizing'

export function planSizingLaptop(windowSize: {
  width: number
  height: number
}): PlanSizing {
  const pad = 30
  const topFn = (transition: number) =>
    linearFnFomPoints(0, 50, 1, 70)(transition)

  const headingMarginBottom = 20

  // Guide
  const guide = (transition: number) => {
    const y = 0
    const height = windowSize.height - y
    const width = windowSize.width * 0.28
    const x = linearFnFomPoints(0, -width + pad * 0.75, 1, 0)(transition)
    return {
      position: rectExt({width, x, height, y}),
      padding: {left: pad, right: pad * 0.75, top: topFn(1), bottom: pad},
      headingMarginBottom,
    }
  }

  // ---- INPUT SUMMARY ----
  const inputSummary = ((): PlanSizing['inputSummary'] => ({
    dynamic: (transition: number) => ({
      region: rectExt({
        x: guide(transition).position.right,
        y: 0,
        height: windowSize.height,
        width: Math.max(windowSize.width * 0.37, 350),
      }),
    }),
    fixed: {
      padding: {
        left: pad * 0.25,
        right: pad * 0.75,
        top: topFn(0),
        bottom: pad,
      },
      cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
    },
  }))()

  // ---- INPUT ----
  const input = ((): PlanSizing['input'] => {
    return {
      dynamic: (transition: number) => ({
        region: rectExt({
          y: topFn(1),
          bottom: windowSize.height,
          width: Math.max(windowSize.width * 0.3, 350),
          x: guide(transition).position.right,
        }),
      }),
      fixed: {
        padding: {left: pad * 0.25, right: pad * 0.75, top: 0, bottom: pad},
        cardPadding: {left: 15, right: 15, top: 15, bottom: 15},
        headingMarginBottom,
      },
    }
  })()

  // Heading
  const heading = (transition: number) => {
    return {
      position: rectExt({
        width: input.dynamic(1).region.right - pad,
        height: 60,
        x: guide(transition).position.x + pad,
        y: 0,
      }),
    }
  }

  // Chart
  const chart = (transition: number) => {
    const padTop = 15
    const positionFn = (transition: number) => {
      const y = topFn(transition) - padTop
      const x =
        linearFnFomPoints(
          0,
          inputSummary.dynamic(0).region.right,
          1,
          input.dynamic(1).region.right
        )(transition) +
        pad * 0.25
      return {
        y,
        x,
        width: windowSize.width - x - pad,
        height: windowSize.height - 2 * y,
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
      padding: {left: 20, right: 20, top: 10, bottom: 10},
      // 18px is text-lg 20px is text-xl.
      menuButtonScale: linearFnFomPoints(0, 1, 1, 18 / 20)(transition),
      cardPadding: {left: 15, right: 15, top: 10, bottom: 10},
      headingMarginBottom: 10,
      legacyWidth: linearFnFomPoints(0, 120, 1, 100)(transition),
      intraGap: linearFnFomPoints(0, 20, 1, 10)(transition),
    }
  }

  return {input, inputSummary, guide, chart, heading}
}
