import {Power1, Power4} from 'gsap'
import {useRouter} from 'next/router'
import {useCallback, useEffect, useMemo, useState} from 'react'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {useURLParam} from '../../../Utils/UseURLParam'
import {assert, noCase} from '../../../Utils/Utils'
import {SimulationInfo, useSimulation} from '../../App/WithSimulation'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
  isChartPanelType,
} from './ChartPanelType'
import {TPAWChartState} from './TPAWChart'
import {
  tpawChartData,
  tpawChartDataScaled,
  tpawChartDataYRange,
} from './TPAWChartData'
import {
  tpawChartLegacyData,
  tpawChartLegacyDataYRange,
} from './TPAWChartLegacyData'

type _State = {
  type: ChartPanelType
} & TPAWChartState

const morphAnimation: ChartAnimation = {ease: Power4.easeOut, duration: 1.5 }
const normalAnimation: ChartAnimation = {ease: Power1.easeOut, duration: 0.5 }

export type ChartPanelState = ReturnType<typeof useChartPanelState>
export function useChartPanelState() {
  const {tpawResult, highlightPercentiles} = useSimulation()
  const {params} = tpawResult.args

  const router = useRouter()

  const panelTypeInStr = useURLParam('view') ?? ''
  const panelTypeIn = isChartPanelType(params, panelTypeInStr)
    ? panelTypeInStr
    : 'spending-total'

  const [state, setState] = useState<_State>(() => {
    const type = panelTypeIn
    const data = tpawChartData(type, tpawResult, highlightPercentiles)
    const legacyData = tpawChartLegacyData(tpawResult, highlightPercentiles)
    return {
      type,
      main: {
        data,
        yRange: tpawChartDataYRange(data),
      },
      legacy: {
        data: legacyData,
        yRange: tpawChartLegacyDataYRange(legacyData),
        show: _shouldShowLegacy(type, tpawResult),
      },
      animation: null,
    }
  })

  const [yAxisFormat] = useMemo(() => _info(state.type), [state.type])

  useEffect(() => {
    setState(prev => {
      const type = prev.type
      const show = _shouldShowLegacy(type, tpawResult)
      const legacyData = tpawChartLegacyData(tpawResult, highlightPercentiles)
      return {
        type,
        main: {
          data: tpawChartData(type, tpawResult, highlightPercentiles),
          yRange: prev.main.yRange,
        },
        legacy: {
          data: legacyData,
          yRange:
            show && !prev.legacy.show
              ? tpawChartLegacyDataYRange(legacyData)
              : prev.legacy.yRange,
          show,
        },
        animation: normalAnimation,
      }
    })
  }, [tpawResult, highlightPercentiles])
  useAssertConst([highlightPercentiles])

  const handleRescale = useCallback(() => {
    setState(prev => {
      return {
        type: prev.type,
        main: {
          data: prev.main.data,
          yRange: tpawChartDataYRange(prev.main.data),
        },
        legacy: {
          data: prev.legacy.data,
          yRange: tpawChartLegacyDataYRange(prev.legacy.data),
          show: prev.legacy.show,
        },
        animation: normalAnimation,
      }
    })
  }, [])

  const handleChangeType = (type: ChartPanelType) => {
    if (type === state.type) return // this can be caused by update from url., eg back button.
    const mainYRange = tpawChartDataYRange(
      tpawChartData(type, tpawResult, highlightPercentiles)
    )
    const url = new URL(window.location.href)
    if (type === 'spending-total') {
      url.searchParams.delete('view')
    } else {
      url.searchParams.set('view', type)
    }
    void router.push(url)

    setState(prev => ({
      type,
      main: {
        data: tpawChartDataScaled(prev.main.data, mainYRange),
        yRange: mainYRange,
      },
      legacy: {
        data: prev.legacy.data,
        yRange: prev.legacy.yRange,
        show: prev.legacy.show,
      },
      animation: null,
    }))
  }

  // THis is needed for the browser back button to work.
  useEffect(() => {
    if (panelTypeIn !== state.type) handleChangeType(panelTypeIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panelTypeIn])

  useEffect(() => {
    setState(prev => {
      assert(prev.type === state.type)
      return {
        type: prev.type,
        main: {
          data: tpawChartData(prev.type, tpawResult, highlightPercentiles),
          yRange: prev.main.yRange,
        },
        legacy: {
          data: prev.legacy.data,
          yRange: prev.legacy.yRange,
          show: _shouldShowLegacy(prev.type, tpawResult),
        },
        animation: morphAnimation,
      }
    })
    // exclude tpawResult.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightPercentiles, state.type])
  useAssertConst([highlightPercentiles])

  return {state, handleChangeType, yAxisFormat, handleRescale}
}

const _info = (type: ChartPanelType): [(x: number) => string] => {
  switch (type) {
    case 'spending-total':
    case 'spending-general':
      return [formatCurrency]
    case 'portfolio':
      return [formatCurrency]
    case 'glide-path':
      return [formatPercentage(0)]
    case 'withdrawal-rate':
      return [formatPercentage(1)]
    default:
      if (isChartPanelSpendingEssentialType(type)) return [formatCurrency]
      if (isChartPanelSpendingDiscretionaryType(type)) return [formatCurrency]
      noCase(type)
  }
}

const _shouldShowLegacy = (
  type: ChartPanelType,
  {args}: SimulationInfo['tpawResult']
) => {
  const hasLegacy =
    args.params.legacy.total > 0 || args.params.spendingCeiling !== null
  const legacyMakesSense = (() => {
    switch (type) {
      case 'spending-total':
      case 'spending-general':
      case 'portfolio':
        return true
      case 'glide-path':
      case 'withdrawal-rate':
        return false
      default:
        if (isChartPanelSpendingEssentialType(type)) return true
        if (isChartPanelSpendingDiscretionaryType(type)) return true
        noCase(type)
    }
  })()
  return hasLegacy && legacyMakesSense
}
