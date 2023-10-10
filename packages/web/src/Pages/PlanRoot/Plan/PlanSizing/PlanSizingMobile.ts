import { block, linearFnFomPoints, noCase } from '@tpaw/common'
import { Size, newPadding, rectExt } from '../../../../Utils/Geometry'
import { headerHeight } from '../../../App/Header'
import { PlanSizing } from './PlanSizing'

const cardPadding = { left: 15, right: 15, top: 10, bottom: 10 }

export function planSizingMobile(
  windowSize: Size,
  scrollbarWidth: number,
  isSWR: boolean,
  isTallMenu: boolean,
): PlanSizing {
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
    ): PlanSizing['input']['dynamic']['dialogIn'] => ({
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
        top: pad * 4 - 16 + (isTallMenu ? 30 : 0), // -16 to account for py-4 on PlanInputBodyHeader
      },
    })
    return {
      dynamic: {
        dialogIn: dynamic('dialog', 'Input'),
        dialogOut: dynamic('dialog', 'Summary'),
        notDialogIn: dynamic('notDialog', 'Input'),
        notDialogOut: dynamic('notDialog', 'Summary'),
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
        horz: pad,
        top: pad * 4 + (isTallMenu ? 30 : 0),
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
        in: input.dynamic.notDialogIn,
        out: input.dynamic.notDialogOut,
      },
      fixed: input.fixed.notDialogMode,
    }
  })()

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
        right: windowSize.width - 20,
        bottom: windowSize.height - 20,
      },
      opacity: 0,
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
  return { input, help, chart, summary, menu, contact }
}
