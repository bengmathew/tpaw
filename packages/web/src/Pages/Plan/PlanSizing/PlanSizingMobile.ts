import { linearFnFomPoints } from '@tpaw/common'
import { newPadding, rectExt, Size } from '../../../Utils/Geometry'
import { headerHeight } from '../../App/Header'
import { PlanSizing } from './PlanSizing'

const cardPadding = { left: 15, right: 15, top: 10, bottom: 10 }

export function planSizingMobile(windowSize: Size, isSWR: boolean): PlanSizing {
  const pad = windowSize.width < 400 ? 10 : 12
  // ---- WELCOME ----
  const welcome = ((): PlanSizing['welcome'] => {
    const width = windowSize.width - pad * 2
    const inOriginX = (windowSize.width - width) / 2
    return {
      dynamic: {
        in: { origin: { x: inOriginX, y: headerHeight }, opacity: 1 },
        out: { origin: { x: inOriginX - 25, y: headerHeight }, opacity: 0 },
      },
      fixed: {
        size: { width, height: windowSize.height - headerHeight },
      },
    }
  })()

  // ---- CHART ----
  const chart = ((): PlanSizing['chart'] => {
    type Dynamic = PlanSizing['chart']['dynamic']['hidden']
    const summaryState: Dynamic = {
      region: rectExt({
        x: 0,
        y: 0,
        width: windowSize.width,
        height:
          Math.min(
            400,
            linearFnFomPoints(375, 355, 412, 360)(windowSize.width),
          ) + (isSWR ? 15 : 0),
      }),
      padding: newPadding({
        top: pad * 0.5 + headerHeight,
        bottom: 45,
        horz: pad,
      }),
      borderRadius: 0,
      opacity: 1,
    }

    const inputState = {
      ...summaryState,
    }

    const resultsState: Dynamic = inputState
    const hiddenState = {
      ...resultsState,
      region: rectExt.translate(resultsState.region, { x: 0, y: -15 }),
      opacity: 0,
    }
    return {
      dynamic: {
        summary: summaryState,
        input: inputState,
        results: resultsState,
        hidden: hiddenState,
      },
      fixed: {
        intraGap: pad,
        cardPadding: { top: 7, left: 10, right: 10, bottom: 7 },
      },
    }
  })()

  // ---- INPUT ----
  const input = (() => {
    const sizing: PlanSizing['input'] = {
      dynamic: {
        dialogModeIn: {
          origin: { x: 0, y: headerHeight },
          opacity: 1,
        },
        dialogModeOutRight: {
          origin: { x: 25, y: headerHeight },
          opacity: 0,
        },
        dialogModeOutLeft: {
          origin: { x: -25, y: headerHeight },
          opacity: 0,
        },
        notDialogModeIn: {
          origin: { x: 0, y: chart.dynamic.input.region.bottom },
          opacity: 1,
        },
        notDialogModeOut: {
          origin: { x: 15, y: chart.dynamic.summary.region.bottom },
          opacity: 0,
        },
      },
      fixed: {
        dialogMode: {
          size: {
            width: windowSize.width,
            height: windowSize.height - headerHeight,
          },
          padding: { horz: pad, top: pad * 3 },
        },
        notDialogMode: {
          size: {
            width: windowSize.width,
            height: windowSize.height - chart.dynamic.input.region.bottom,
          },
          padding: { horz: pad, top: pad * 3 },
        },
        cardPadding,
      },
    }
    return sizing
  })()

  // ---- SUMMARY ----
  const summary = ((): PlanSizing['summary'] => {
    return {
      dynamic: {
        in: {
          origin: { x: 0, y: chart.dynamic.summary.region.bottom },
          opacity: 1,
        },
        out: {
          origin: { x: -15, y: chart.dynamic.input.region.bottom },
          opacity: 0,
        },
      },
      fixed: {
        size: {
          width: windowSize.width,
          height: windowSize.height - chart.dynamic.summary.region.bottom,
        },
        padding: newPadding({ horz: pad, top: pad / 2, bottom: 0 }),
        cardPadding,
      },
    }
  })()

  // ---- RESULTS ----
  const results = ((): PlanSizing['results'] => {
    return {
      dynamic: {
        in: {
          origin: { x: 0, y: chart.dynamic.results.region.bottom },
          opacity: 1,
        },
        outDialogMode: {
          origin: { x: 0, y: chart.dynamic.results.region.bottom + 15 },
          opacity: 0,
        },
        outNotDialogMode: {
          origin: {
            x: input.dynamic.notDialogModeOut.origin.x,
            y: chart.dynamic.summary.region.bottom,
          },
          opacity: 0,
        },
      },
      fixed: {
        size: {
          width: windowSize.width,
          height: windowSize.height - chart.dynamic.results.region.bottom,
        },
        padding: newPadding(pad),
      },
    }
  })()

  return { welcome, chart, input, results, summary }
}
