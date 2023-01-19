import { newPadding, rectExt, Size } from '../../../Utils/Geometry'
import { PlanSizing } from './PlanSizing'

const pad = 40
const cardPadding = newPadding(20)

export function planSizingDesktop(
  windowSize: Size,
  isSWR: boolean,
): PlanSizing {
  const contentWidth = 600

  // ---- CHART ----
  const chart = ((): PlanSizing['chart'] => {
    const dynamic = ({
      dialog,
      input,
    }: {
      dialog: boolean
      input: boolean
    }): PlanSizing['chart']['dynamic']['dialogInput'] => {
      const baseHeight = 495 + 30
      const height = baseHeight * (dialog ? 0.75 : 1) - (input ? 30 : 0)

      const basePaddingVert = pad * 1.5
      return {
        region: rectExt({ x: 0, y: 0, width: windowSize.width, height }),
        padding: newPadding({
          top: basePaddingVert,
          horz: pad,
          bottom: basePaddingVert * (input ? 0.9 : 1),
        }),
        borderRadius: 0,
        opacity: 1,
      }
    }

    return {
      dynamic: {
        dialogSummary: dynamic({ dialog: true, input: false }),
        dialogInput: dynamic({ dialog: true, input: true }),
        notDialogSummary: dynamic({ dialog: false, input: false }),
        notDialogInput: dynamic({ dialog: false, input: true }),
      },
      fixed: {
        intraGap: pad / 2,
        cardPadding: { left: 15, right: 15, top: 10, bottom: 10 },
      },
    }
  })()

  // ---- INPUT ----
  const input = ((): PlanSizing['input'] => {
    const dynamic = (
      dialog: 'dialog' | 'notDialog',
      section: 'Summary' | 'Input',
    ): PlanSizing['input']['dynamic']['dialogModeIn'] => ({
      origin: { x: 0, y: chart.dynamic[`${dialog}${section}`].region.bottom },
      opacity: section === 'Input' ? 1 : 0,
    })
    const fixed = (
      dialog: 'dialog' | 'notDialog',
    ): PlanSizing['input']['fixed']['dialogMode'] => ({
      size: {
        width: windowSize.width,
        height:
          windowSize.height - chart.dynamic[`${dialog}Input`].region.bottom,
      },
      padding: {
        left: pad,
        right: windowSize.width - contentWidth - pad,
        top: pad - 16, // -16 to account for py-4 on PlanInputBodyHeader
      },
    })
    return {
      dynamic: {
        dialogModeIn: dynamic('dialog', 'Input'),
        dialogModeOut: dynamic('dialog', 'Summary'),
        notDialogModeIn: dynamic('notDialog', 'Input'),
        notDialogModeOut: dynamic('notDialog', 'Summary'),
      },
      fixed: {
        dialogMode: fixed('dialog'),
        notDialogMode: fixed('notDialog'),
        cardPadding,
      },
    }
  })()

  // ---- SUMMARY ----
  const summary = ((): PlanSizing['summary'] => {
    const dynamic = (
      dialog: 'dialog' | 'notDialog',
      section: 'Summary' | 'Input',
    ): PlanSizing['summary']['dynamic']['dialogIn'] => ({
      origin: { x: 0, y: chart.dynamic[`${dialog}${section}`].region.bottom },
      opacity: section === 'Summary' ? 1 : 0,
    })

    const fixed = (
      dialog: 'dialog' | 'notDialog',
    ): PlanSizing['summary']['fixed']['dialogMode'] => ({
      size: {
        width: windowSize.width,
        height:
          windowSize.height - chart.dynamic[`${dialog}Summary`].region.height,
      },
      padding: newPadding({
        left: pad,
        right: windowSize.width - contentWidth - pad,
        top: pad / 2,
        bottom: 0,
      }),
    })
    return {
      dynamic: {
        dialogIn: dynamic('dialog', 'Summary'),
        dialogOut: dynamic('dialog', 'Input'),
        notDialogIn: dynamic('notDialog', 'Summary'),
        notDialogOut: dynamic('notDialog', 'Input'),
      },
      fixed: {
        dialogMode: fixed('dialog'),
        notDialogMode: fixed('notDialog'),
        cardPadding,
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

  return { chart, input, help, summary }
}
