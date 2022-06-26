import {useCallback, useEffect, useState} from 'react'
import {useAssertConst} from '../../../Utils/UseAssertConst'
import {useURLUpdater} from '../../../Utils/UseRouterPush'
import {useURLParam} from '../../../Utils/UseURLParam'
import {useSimulation} from '../../App/WithSimulation'
import {ChartPanelType, isChartPanelType} from './ChartPanelType'

export function useChartPanelTypeState() {
  const simulation = useSimulation()
  const panelTypeInStr = useURLParam('view') ?? ''
  const panelTypeIn = isChartPanelType(
    simulation.tpawResult.args.params.original,
    panelTypeInStr
  )
    ? panelTypeInStr
    : 'spending-total'

  const [chartPanelType, setChartPanelTypeLocal] = useState<
    ChartPanelType | 'sharpe-ratio'
  >(panelTypeIn)

  const urlUpdater = useURLUpdater()
  const setChartPanelType = useCallback<typeof setChartPanelTypeLocal>(
    x => {
      setChartPanelTypeLocal(prev => {
        const type = typeof x === 'function' ? x(prev) : x
        if (type !== 'sharpe-ratio') {
          const url = new URL(window.location.href)
          if (type === 'spending-total') {
            url.searchParams.delete('view')
          } else {
            url.searchParams.set('view', type)
          }
          void urlUpdater.push(url)
        }
        return type
      })
    },
    [urlUpdater]
  )
  useAssertConst([urlUpdater])

  // This is needed for the browser back button to work.
  useEffect(() => {
    if (panelTypeIn !== chartPanelType) setChartPanelType(panelTypeIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panelTypeIn])

  return [chartPanelType, setChartPanelType] as const
}
