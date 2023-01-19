import { linearFnFomPoints } from '@tpaw/common'
import { newPadding, rectExt, Size } from '../../../Utils/Geometry'
import { headerHeight } from '../../App/Header'
import { PlanSizing } from './PlanSizing'

const cardPadding = { left: 15, right: 15, top: 10, bottom: 10 }

export function planSizingMobile(windowSize: Size, isSWR: boolean): PlanSizing {
  const pad = windowSize.width < 400 ? 10 : 12

  // ---- CHART ----
  const chart = ((): PlanSizing['chart'] => {
    const dynamic = ({
      dialog,
      input,
    }: {
      dialog: boolean
      input: boolean
    }): PlanSizing['chart']['dynamic']['dialogInput'] => {
      const baseHeight =
        Math.min(400, linearFnFomPoints(375, 355, 412, 360)(windowSize.width)) +
        (isSWR ? 15 : 0)
      const height = baseHeight

      return {
        region: rectExt({ x: 0, y: 0, width: windowSize.width, height }),
        padding: newPadding({
          top: pad * 0.5 + headerHeight,
          bottom: 45,
          horz: pad,
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
        intraGap: pad,
        cardPadding: { top: 7, left: 10, right: 10, bottom: 7 },
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
        horz: pad,
        top: pad * 3 - 16, // -16 to account for py-4 on PlanInputBodyHeader
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
      padding: newPadding({ horz: pad, top: pad / 2, bottom: 0 }),
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
