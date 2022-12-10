import { newPadding, rectExt, Size } from '../../../Utils/Geometry'
import { PlanSizing } from './PlanSizing'

export function planSizingLaptop(windowSize: Size,
  isSWR:boolean): PlanSizing {
  const pad = 40

  const summaryWidth = Math.max(windowSize.width * 0.38, 500)

  // ---- WELCOME ----
  const welcome = ((): PlanSizing['welcome'] => {
    const width = 500
    const inOriginX = (windowSize.width - width) / 2
    return {
      dynamic: {
        in: { origin: { x: inOriginX, y: 0 }, opacity: 1 },
        out: { origin: { x: inOriginX - 25, y: 0 }, opacity: 0 },
      },
      fixed: {
        size: {
          width,
          height: windowSize.height,
        },
      },
    }
  })()

  // ---- INPUT ----
  const input = ((): PlanSizing['input'] => {
    const inputWidth = {
      dialogMode: 500,
      notDialogMode: summaryWidth * 1.15,
    }
    type Dynamic = PlanSizing['input']['dynamic']['dialogModeIn']
    const dialogModeIn: Dynamic = {
      origin: { x: 0, y: 0 },
      opacity: 1,
    }
    return {
      dynamic: {
        dialogModeIn,
        dialogModeOutRight: {
          origin: { x: 25, y: 0 },
          opacity: 0,
        },
        dialogModeOutLeft: {
          origin: { x: -25, y: 0 },
          opacity: 0,
        },
        notDialogModeIn: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        notDialogModeOut: {
          origin: { x: -(inputWidth.notDialogMode - summaryWidth), y: 0 },
          opacity: 0,
        },
      },
      fixed: {
        dialogMode: {
          size: { ...windowSize },
          padding: {
            horz: (windowSize.width - inputWidth.dialogMode) / 2,
            top: pad * 2,
          },
        },
        notDialogMode: {
          size: {
            width: inputWidth.notDialogMode,
            height: windowSize.height,
          },
          padding: { left: pad, right: pad * 0.75, top: pad * 2 },
        },
        cardPadding: newPadding(20),
      },
    }
  })()

  // ---- RESULTS ----
  const results = ((): PlanSizing['results'] => {
    const width = input.fixed.notDialogMode.size.width
    return {
      dynamic: {
        in: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        outDialogMode: {
          // origin: {x: -(width - summaryWidth), y: 0},

          origin: { x: 50, y: 0 },
          opacity: 0,
        },
        outNotDialogMode: {
          origin: { x: -50, y: 0 },
          opacity: 0,
        },
      },
      fixed: {
        size: { width, height: windowSize.height },
        padding: { left: pad, right: pad * 0.95, top: pad * 2 },
      },
    }
  })()

  // ---- SUMMARY ----
  const summary = ((): PlanSizing['summary'] => {
    const size = { width: summaryWidth, height: windowSize.height }

    return {
      dynamic: {
        in: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        out: {
          origin: {
            x:
              input.dynamic.notDialogModeIn.origin.x +
              input.fixed.notDialogMode.size.width -
              size.width,
            y: 0,
          },
          opacity: 0,
        },
      },
      fixed: {
        size,
        padding: { left: pad, right: pad * 0.75, top: pad / 2 },
        cardPadding: newPadding(20),
      },
    }
  })()

  // ---- CHART ----
  const chart = ((): PlanSizing['chart'] => {
    type Dynamic = PlanSizing['chart']['dynamic']['hidden']
    const cardPadding = { left: 15, right: 15, top: 10, bottom: 10 }
    const exceptRegionAndFullyVisible: Omit<Dynamic, 'region'> = {
      padding: newPadding({ vert: pad * 1.5, horz: pad }),
      borderRadius: 16,
      opacity: 1,
    }
    const summaryState: Dynamic = (() => {
      return {
        ...exceptRegionAndFullyVisible,
        region: rectExt({
          x:
            summary.dynamic.in.origin.x + summary.fixed.size.width + pad * 0.25,
          y: pad * 2,
          right: windowSize.width - pad,
          bottom: windowSize.height - pad * 2,
        }),
      }
    })()
    const inputState = {
      ...exceptRegionAndFullyVisible,
      region: rectExt({
        x:
          input.dynamic.notDialogModeIn.origin.x +
          input.fixed.notDialogMode.size.width +
          pad * 0.25,
        y: summaryState.region.y,
        right: windowSize.width - pad,
        bottom: windowSize.height - pad * 4,
      }),
      padding: newPadding({
        horz: summaryState.padding.left * 0.85,
        vert: summaryState.padding.top * 0.85,
      }),
    }
    const resultsState = inputState
    const hiddenState = {
      ...resultsState,
      region: rectExt.translate(resultsState.region, { x: 50, y: 0 }),
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
        intraGap: 20,
        cardPadding,
      },
    }
  })()

  return { welcome, input, results, chart, summary }
}
