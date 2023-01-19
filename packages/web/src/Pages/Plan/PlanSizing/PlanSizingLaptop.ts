import { newPadding, rectExt, Size } from '../../../Utils/Geometry'
import { PlanSizing } from './PlanSizing'

export function planSizingLaptop(windowSize: Size, isSWR: boolean): PlanSizing {
  const pad = 40

  const summaryWidth = Math.max(windowSize.width * 0.38, 500)

  // ---- INPUT ----
  const input = ((): PlanSizing['input'] => {
    const inputWidth = summaryWidth * 1.15

    const fixed = {
      size: {
        width: inputWidth,
        height: windowSize.height,
      },
      padding: {
        left: pad,
        right: pad * 0.75,
        top: pad * 2 - 16, // -16 to account for py-4 on PlanInputBodyHeader,
      },
    }

    return {
      dynamic: {
        dialogModeIn: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        dialogModeOut: {
          origin: { x: -(inputWidth - summaryWidth), y: 0 },
          opacity: 0,
        },
        notDialogModeIn: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        notDialogModeOut: {
          origin: { x: -(inputWidth - summaryWidth), y: 0 },
          opacity: 0,
        },
      },
      fixed: {
        dialogMode: fixed,
        notDialogMode: fixed,
        cardPadding: newPadding(20),
      },
    }
  })()

  // ---- HELP ----
  const help = ((): PlanSizing['help'] => {
    return {
      dynamic: {
        in: input.dynamic.notDialogModeIn,
        out: input.dynamic.notDialogModeOut,
      },
      fixed: input.fixed.notDialogMode,
    }
  })()

  // ---- SUMMARY ----
  const summary = ((): PlanSizing['summary'] => {
    const size = { width: summaryWidth, height: windowSize.height }

    type Dynamic = PlanSizing['summary']['dynamic']['dialogIn']
    const dialogIn: Dynamic = {
      origin: { x: 0, y: 0 },
      opacity: 1,
    }
    const dialogOut: Dynamic = {
      origin: {
        x:
          input.dynamic.notDialogModeIn.origin.x +
          input.fixed.notDialogMode.size.width -
          size.width,
        y: 0,
      },
      opacity: 0,
    }
    const fixedDialog = {
      size,
      padding: { left: pad, right: pad * 0.75, top: pad / 2 },
    }
    return {
      dynamic: {
        dialogIn,
        dialogOut,
        notDialogIn: dialogIn,
        notDialogOut: dialogOut,
      },
      fixed: {
        dialogMode: fixedDialog,
        notDialogMode: fixedDialog,
        cardPadding: newPadding(20),
      },
    }
  })()

  // ---- CHART ----
  const chart = ((): PlanSizing['chart'] => {
    type Dynamic = PlanSizing['chart']['dynamic']['dialogInput']
    const cardPadding = { left: 15, right: 15, top: 10, bottom: 10 }
    const exceptRegionAndFullyVisible: Omit<Dynamic, 'region'> = {
      padding: newPadding({ vert: pad * 1.5, horz: pad }),
      borderRadius: 16,
      opacity: 1,
    }
    const notDialogSummary: Dynamic = (() => {
      return {
        ...exceptRegionAndFullyVisible,
        region: rectExt({
          x:
            summary.dynamic.notDialogIn.origin.x +
            summary.fixed.notDialogMode.size.width +
            pad * 0.25,
          y: pad * 2,
          right: windowSize.width - pad,
          bottom: windowSize.height - pad * 2,
        }),
      }
    })()
    const notDialogInput = {
      ...exceptRegionAndFullyVisible,
      region: rectExt({
        x:
          input.dynamic.notDialogModeIn.origin.x +
          input.fixed.notDialogMode.size.width +
          pad * 0.25,
        y: notDialogSummary.region.y,
        right: windowSize.width - pad,
        bottom: windowSize.height - pad * 4,
      }),
      padding: newPadding({
        horz: notDialogSummary.padding.left * 0.85,
        vert: notDialogSummary.padding.top * 0.85,
      }),
    }
    // const resultsState = notDialogInput
    // const hiddenState = {
    //   ...resultsState,
    //   region: rectExt.translate(resultsState.region, { x: 50, y: 0 }),
    //   opacity: 0,
    // }
    return {
      dynamic: {
        dialogSummary: notDialogSummary,
        dialogInput: notDialogInput,
        notDialogSummary,
        notDialogInput,
      },
      fixed: {
        intraGap: 20,
        cardPadding,
      },
    }
  })()

  return { input, help, chart, summary }
}
