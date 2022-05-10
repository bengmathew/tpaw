import {Power1, Power4} from 'gsap'
import {useRouter} from 'next/router'
import {useCallback, useEffect, useState} from 'react'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {useURLParam} from '../../../Utils/UseURLParam'
import {assert} from '../../../Utils/Utils'
import {SimulationInfo, useSimulation} from '../../App/WithSimulation'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {ChartReactState} from '../../Common/Chart/ChartReact'
import {ChartPanelType, isChartPanelType} from './ChartPanelType'
import {
  TPAWChartDataLegacy,
  tpawChartDataLegacy,
  tpawChartDataLegacyYRange,
} from './TPAWChart/TPAWChartDataLegacy'
import {
  TPAWChartDataMain,
  tpawChartDataMain,
  tpawChartDataMainYRange,
  tpawChartDataScaled,
} from './TPAWChart/TPAWChartDataMain'

type _State = {
  type: ChartPanelType
  main: ChartReactState<TPAWChartDataMain>
  legacy: ChartReactState<TPAWChartDataLegacy>
}

const morphAnimation: ChartAnimation = {ease: Power4.easeOut, duration: 1.5}
const normalAnimation: ChartAnimation = {ease: Power1.easeOut, duration: 0.5}

export type ChartPanelState = ReturnType<typeof useChartPanelState>
export function useChartPanelState() {
  const {tpawResult, highlightPercentiles} = useSimulation()
  const {params} = tpawResult.args

  const router = useRouter()

  const panelTypeInStr = useURLParam('view') ?? ''
  const panelTypeIn = isChartPanelType(params.original, panelTypeInStr)
    ? panelTypeInStr
    : 'spending-total'

  const [state, setState] = useState<_State>(() => {
    const type = panelTypeIn
    const data = tpawChartDataMain(type, tpawResult, highlightPercentiles)
    const animation = null
    const legacyData = tpawChartDataLegacy(tpawResult, highlightPercentiles)
    return {
      type,
      main: {
        data,
        xyRange: {x: data.years.displayRange, y: tpawChartDataMainYRange(data)},
        animation,
      },
      legacy: {
        data: legacyData,
        xyRange: {
          x: {start: 0, end: 1},
          y: tpawChartDataLegacyYRange(legacyData),
        },
        animation,
      },
    }
  })

  useEffect(() => {
    setState(prev => {
      const type = prev.type
      const legacyData = tpawChartDataLegacy(tpawResult, highlightPercentiles)
      const data = tpawChartDataMain(type, tpawResult, highlightPercentiles)
      const animation = normalAnimation
      return {
        type,
        main: {
          data,
          xyRange: {x: data.years.displayRange, y: prev.main.xyRange.y},
          animation,
        },
        legacy: {
          data: legacyData,
          xyRange: {
            x: {start: 0, end: 1},
            y: prev.legacy.xyRange.y,
          },
          animation,
        },
      }
    })
  }, [tpawResult, highlightPercentiles])
  useAssertConst([highlightPercentiles])

  const handleRescale = useCallback(() => {
    setState(prev => {
      const data = prev.main.data
      const animation = normalAnimation
      return {
        type: prev.type,
        main: {
          data: data,
          xyRange: {
            x: data.years.displayRange,
            y: tpawChartDataMainYRange(prev.main.data),
          },
          animation,
        },
        legacy: {
          data: prev.legacy.data,
          xyRange: {
            x: {start: 0, end: 1},
            y: tpawChartDataLegacyYRange(prev.legacy.data),
          },
          animation,
        },
      }
    })
  }, [])

  const handleChangeType = (type: ChartPanelType) => {
    if (type === state.type) return // this can be caused by update from url., eg back button.
    const url = new URL(window.location.href)
    if (type === 'spending-total') {
      url.searchParams.delete('view')
    } else {
      url.searchParams.set('view', type)
    }
    void router.push(url)
    const animation = null

    setState(prev => {
      const yRange = tpawChartDataMainYRange(
        tpawChartDataMain(type, tpawResult, highlightPercentiles)
      )
      const data = tpawChartDataScaled(prev.main.data, yRange)
      const xyRange = {x: data.years.displayRange, y: yRange}

      return {
        type,
        main: {data, xyRange, animation},
        legacy: {
          data: prev.legacy.data,
          xyRange: {x: {start: 0, end: 1}, y: prev.legacy.xyRange.y},
          animation,
        },
      }
    })
  }

  // THis is needed for the browser back button to work.
  useEffect(() => {
    if (panelTypeIn !== state.type) handleChangeType(panelTypeIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panelTypeIn])

  useEffect(() => {
    setState(prev => {
      assert(prev.type === state.type)
      const animation = morphAnimation
      const data = tpawChartDataMain(
        prev.type,
        tpawResult,
        highlightPercentiles
      )
      return {
        type: prev.type,
        main: {
          data,
          xyRange: {x: data.years.displayRange, y: prev.main.xyRange.y},
          animation,
        },
        legacy: {
          data: prev.legacy.data,
          xyRange: {x: {start: 0, end: 1}, y: prev.legacy.xyRange.y},
          animation,
        },
      }
    })
    // exclude tpawResult.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightPercentiles, state.type])
  useAssertConst([highlightPercentiles])

  const shouldShowLegacy = _shouldShowLegacy(state.type, tpawResult)
  useEffect(() => {
    if (shouldShowLegacy) {
      setState(prev => ({
        ...prev,
        legacy: {
          data: prev.legacy.data,
          xyRange: {
            x: {start: 0, end: 1},
            y: tpawChartDataLegacyYRange(prev.legacy.data),
          },
          animation: normalAnimation,
        },
      }))
    }
  }, [shouldShowLegacy])
  return {state, handleChangeType, handleRescale, shouldShowLegacy}
}

const _shouldShowLegacy = (
  type: ChartPanelType,
  {args}: SimulationInfo['tpawResult']
) => {
  const hasLegacy =
    args.params.legacy.total > 0 || args.params.spendingCeiling !== null
  // const legacyMakesSense = (() => {
  //   switch (type) {
  //     case 'spending-total':
  //     case 'spending-general':
  //     case 'portfolio':
  //       return true
  //     case 'glide-path':
  //     case 'withdrawal-rate':
  //       return false
  //     default:
  //       if (isChartPanelSpendingEssentialType(type)) return true
  //       if (isChartPanelSpendingDiscretionaryType(type)) return true
  //       noCase(type)
  //   }
  // })()
  const legacyMakesSense = true
  return hasLegacy && legacyMakesSense
}
