import { block } from '@tpaw/common'
import { useMemo } from 'react'
import { indigo } from '../../../Utils/ColorPalette'
import { RGB, makeShades } from '../../../Utils/ColorUtils'
import { Record } from '../../../Utils/Record'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'

export const usePlanColors = () => {
  const { simulationInfoByMode } = useSimulation()

  return useMemo(
    () => _planColors[`mainPlan`](simulationInfoByMode.mode),
    [simulationInfoByMode.mode],
  )
}

type RGBAndHex = { rgb: RGB; hex: string }
export type PlanColors = {
  shades: Record<'main' | 'alt' | 'light', RGBAndHex[]>
  dark: RGBAndHex
  fgForDarkBG: RGBAndHex
  pageBG: string
  summaryButtonOuter: (isDialogMode: boolean) => string
  results: {
    bg: string
    fg: string
    darkBG: string
    fgForDarkBG: string
    cardBG: string
  }
}

const _planColors = {
  mainPlan: (mode: 'history' | 'plan'): PlanColors => {
    const pageBG = 'bg-gray-100'
    const cardBG = 'bg-[rgba(255,255,255,.95)]'

    const shades = block(() => {
      const main = makeShades(mode === 'plan' ? indigo[800] : indigo[950])
      const light = makeShades(main[3])
      const justHex = {
        main,
        light,
        alt: makeShades(makeShades(light[5])[11]),
      }
      return Record.mapValues(justHex, (x) =>
        x.map((hex) => ({ hex, rgb: RGB.fromHex(hex) })),
      )
    })

    const dark = shades.main[14]
    const fgForDarkBG = shades.main[2]
    return {
      shades,
      dark,
      fgForDarkBG: fgForDarkBG,
      pageBG,
      summaryButtonOuter: (isDialogMode) =>
        isDialogMode ? _cn('bg-orange-50') : cardBG,

      results: block(() => {
        const fg = shades.main[14].hex
        return {
          bg: shades.alt[6].hex,
          cardBG: shades.light[1].hex,
          fg: fg,
          darkBG: dark.hex,
          fgForDarkBG: fgForDarkBG.hex,
        }
      }),
    }
  },
}

// To help with tailwind CSS auto completion.
// Add "_cn\\('([^('\\)]*)" to tailwindCSS.experimental.classRegex setting in
// vscode workspace settings.json.
const _cn = (className: string) => className

export const mainPlanColors = _planColors.mainPlan('plan')
