import {faInfo} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Power4, Power1} from 'gsap'
import {useRouter} from 'next/router'
import React, {useCallback, useEffect, useMemo, useState} from 'react'
import Measure from 'react-measure'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {useURLParam} from '../../../Utils/UseURLParam'
import {assert, fGet, noCase} from '../../../Utils/Utils'
import {useWindowSize} from '../../../Utils/WithWindowSize'
import {useSimulation} from '../../App/WithSimulation'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {ChartPanelButtons} from './ChartPanelButtons'
import {ChartPanelDescription} from './ChartPanelDescription'
import {ChartPanelMenu} from './ChartPanelMenu'
import {
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
  isChartPanelType,
} from './ChartPanelType'
import {LegacyDisplay} from './LegacyDisplay'
import {TPAWChart, TPAWChartState} from './TPAWChart'
import {
  tpawChartData,
  tpawChartDataScaled,
  tpawChartDataYRange,
} from './TPAWChartData'

type _State = {
  type: ChartPanelType
} & TPAWChartState

const morphAnimation: ChartAnimation = {ease: Power4.easeOut, duration: 1.5}
const normalAnimation: ChartAnimation = {ease: Power1.easeOut, duration: 1}

export function useChartPanel({
  className = '',
  isPortrait,
}: {
  className?: string
  isPortrait: boolean
}) {
  const {tpawResult, highlightPercentiles} = useSimulation()
  const {params} = tpawResult.args

  const [titleWidth, setTitleWidth] = useState(0)
  const [headingSize, setHeadingSize] = useState<{
    width: number
    height: number
  }>({width: 0, height: 100})
  const windowSize = useWindowSize()
  const router = useRouter()
  const topPadding = headingSize.height

  const panelTypeInStr = useURLParam('view') ?? ''
  const panelTypeIn = isChartPanelType(params, panelTypeInStr)
    ? panelTypeInStr
    : 'spending-total'

  const [state, setState] = useState<_State>(() => {
    const type = panelTypeIn
    const data = tpawChartData(type, tpawResult, highlightPercentiles)
    return {
      type,
      data,
      yRange: tpawChartDataYRange(data),
      externalTopPadding: topPadding,
      animation: null,
    }
  })

  const [yAxisFormat, lastAgeIsLegacy] = useMemo(
    () => _info(state.type),
    [state.type]
  )

  useEffect(() => {
    setState(prev => ({
      type: prev.type,
      data: tpawChartData(prev.type, tpawResult, highlightPercentiles),
      yRange: prev.yRange,
      externalTopPadding: prev.externalTopPadding,
      animation: normalAnimation,
    }))
    // HighlighPercentiles is a const.
  }, [tpawResult, highlightPercentiles])

  const handleRescale = useCallback(() => {
    setState(prev => {
      return {
        type: prev.type,
        data: prev.data,
        yRange: tpawChartDataYRange(prev.data),
        externalTopPadding: prev.externalTopPadding,
        animation: normalAnimation,
      }
    })
  }, [])

  const handleChangeType = (type: ChartPanelType) => {
    if (type === state.type) return // this can be caused by update from url., eg back button.
    const yRange = tpawChartDataYRange(
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
      data: tpawChartDataScaled(prev.data, yRange),
      yRange,
      externalTopPadding: prev.externalTopPadding,
      animation: null,
    }))
  }

  // THis is needed for the browser back button to work.
  useEffect(() => {
    handleChangeType(panelTypeIn)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panelTypeIn])

  useEffect(() => {
    setState(prev => {
      assert(prev.type === state.type)
      return {
        type: prev.type,
        data: tpawChartData(prev.type, tpawResult, highlightPercentiles),
        yRange: prev.yRange,
        externalTopPadding: prev.externalTopPadding,
        animation: morphAnimation,
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.type])

  useEffect(() => {
    setState(prev => {
      assert(prev.type === state.type)
      return {
        type: prev.type,
        data: prev.data,
        yRange: prev.yRange,
        externalTopPadding: topPadding,
        // the challenge here is that this can happend during graph change and
        // resize, so not clear if morphAnimation or normal. altogether ugly hack
        // is to make it morph in both cases.
        animation: morphAnimation, 
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topPadding])

  const [showDescriptionPopUp, setShowDescriptionPopUp] = useState(false)
  const render = (
    <div
      className={`${className} relative pb-2 grid`}
      style={{gridArea: 'chart', grid: '"top" 1fr "bottom" auto/ 1fr'}}
    >
      <Measure bounds onResize={({bounds}) => setHeadingSize(fGet(bounds))}>
        {({measureRef}) => (
          <div
            className="flex flex-col items-start absolute w-full  px-3 z-10 bg-pageBG"
            ref={measureRef}
          >
            <div
              className={`flex justify-between items-start w-full ${
                headingSize.width - titleWidth < 300 || windowSize.width < 640
                  ? 'pt-header'
                  : ''
              }`}
            >
              <Measure
                bounds
                onResize={({bounds}) => setTitleWidth(fGet(bounds).width)}
              >
                {({measureRef}) => (
                  <div className="" ref={measureRef}>
                    <ChartPanelMenu
                      type={state.type}
                      onSelect={handleChangeType}
                    />
                  </div>
                )}
              </Measure>
              <button
                className="sm:hidden mt-2 rounded-full w-[23px] h-[23px] bg-gray-700 text-gray-100 text-[11px]"
                onClick={() => setShowDescriptionPopUp(true)}
              >
                <FontAwesomeIcon icon={faInfo} />
              </button>
            </div>
            <ChartPanelDescription
              type={state.type}
              {...{showDescriptionPopUp, setShowDescriptionPopUp}}
            />
            {!isPortrait && (
              <ChartPanelButtons
                className="w-full"
                isPortrait={isPortrait}
                handleRescale={handleRescale}
              />
            )}
          </div>
        )}
      </Measure>
      <div className="grid" style={{grid: '1fr auto /1fr'}}>
        <TPAWChart
          className=" -mx-3 relative z-0"
          yAxisFormat={yAxisFormat}
          lastAgeIsLegacy={lastAgeIsLegacy}
          animationForBoundsChange={normalAnimation}
          state={state}
        />
        {isPortrait && (
          <ChartPanelButtons
            className=""
            isPortrait={isPortrait}
            handleRescale={handleRescale}
          />
        )}
      </div>

      <LegacyDisplay
        className={`absolute left-3`}
        style={{top: `${headingSize.height + 10}px`}}
      />
    </div>
  )
  return [handleChangeType, state.type, render] as const
}

const _info = (type: ChartPanelType): [(x: number) => string, boolean] => {
  switch (type) {
    case 'spending-total':
    case 'spending-regular':
      return [formatCurrency, false]
    case 'portfolio':
      return [formatCurrency, true]
    case 'glide-path':
      return [formatPercentage(0), false]
    case 'withdrawal-rate':
      return [formatPercentage(1), false]
    default:
      if (isChartPanelSpendingEssentialType(type))
        return [formatCurrency, false]
      if (isChartPanelSpendingDiscretionaryType(type))
        return [formatCurrency, false]
      noCase(type)
  }
}
