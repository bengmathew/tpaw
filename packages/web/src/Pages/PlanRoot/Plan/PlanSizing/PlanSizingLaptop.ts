import { block, noCase } from '@tpaw/common'
import { Size, newPadding, rectExt } from '../../../../Utils/Geometry'
import { PlanSizing } from './PlanSizing'

export function planSizingLaptop(
  windowSize: Size,
  scrollbarWidth: number,
  isSWR: boolean,
  tallPlanMenu: boolean,
): Omit<PlanSizing, 'args'> {
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
        dialogIn: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        dialogOut: {
          origin: { x: -(inputWidth - summaryWidth), y: 0 },
          opacity: 0,
        },
        notDialogIn: {
          origin: { x: 0, y: 0 },
          opacity: 1,
        },
        notDialogOut: {
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
        in: input.dynamic.notDialogIn,
        out: input.dynamic.notDialogOut,
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
          input.dynamic.notDialogIn.origin.x +
          input.fixed.notDialogMode.size.width -
          size.width,
        y: 0,
      },
      opacity: 0,
    }
    const fixedDialog = {
      size,
      padding: { left: pad, right: pad * 0.75, top: pad * 2 },
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
      region: block(() => {
        const y = notDialogSummary.region.y
        const height = Math.max(windowSize.height - y - pad * 4, 490)
        return rectExt({
          x:
            input.dynamic.notDialogIn.origin.x +
            input.fixed.notDialogMode.size.width +
            pad * 0.25,
          y,
          right: windowSize.width - pad,
          bottom: y + height,
        })
      }),
      padding: newPadding({
        horz: notDialogSummary.padding.left * 0.85,
        vert: notDialogSummary.padding.top * 0.85,
      }),
    }
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

  // ---- POINTER ----
  const pointer = block<PlanSizing['pointer']>(() => ({
    fixed: block(() => {
      const fromChart = (
        chart: PlanSizing['chart']['dynamic']['dialogInput'],
      ) => {
        return {
          region: rectExt({ x: 0, y: 0, ...windowSize }),
          chartPosition: {
            bottom: chart.region.bottom - chart.padding.bottom,
            left: chart.region.x + chart.padding.left,
          },
        }
      }
      return {
        summary: fromChart(chart.dynamic.notDialogSummary),
        input: fromChart(chart.dynamic.notDialogInput),
        help: fromChart(chart.dynamic.notDialogInput),
      }
    }),
  }))

  // ---- MENU ----
  const menu = block<PlanSizing['menu']>(() => {
    const dynamic = (
      dialog: 'dialog' | 'notDialog',
      section: 'Summary' | 'Input',
    ): PlanSizing['menu']['dynamic']['summaryDialog'] => {
      const src =
        section === 'Summary'
          ? summary
          : section === 'Input'
          ? input
          : noCase(section)
      const srcFixed = src.fixed[`${dialog}Mode`]
      const srcFixedPaddingRight =
        'horz' in srcFixed.padding
          ? srcFixed.padding.horz
          : srcFixed.padding.right

      return {
        position: {
          right: srcFixed.size.width - srcFixedPaddingRight - scrollbarWidth,
          top: src.dynamic[`${dialog}In`].origin.y,
        },
        opacity: 1,
      }
    }

    return {
      dynamic: {
        summaryDialog: dynamic('dialog', 'Summary'),
        summaryNotDialog: dynamic('notDialog', 'Summary'),
        inputDialog: dynamic('dialog', 'Input'),
        inputNotDialog: dynamic('notDialog', 'Input'),
        help: { ...dynamic('notDialog', 'Input'), opacity: 0 },
      },
    }
  })
  // ---- CONTACT ----
  const contact = ((): PlanSizing['contact'] => {
    const dynamic = {
      position: {
        right: windowSize.width - 40,
        bottom: windowSize.height - 20,
      },
      opacity: 1,
    }

    return {
      dynamic: {
        summaryDialog: dynamic,
        summaryNotDialog: dynamic,
        inputDialog: dynamic,
        inputNotDialog: dynamic,
        help: dynamic,
      },
    }
  })()
  return { input, help, chart, summary, menu, contact, pointer }
}
