import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS, paddingCSSStyleHorz} from '../../../Utils/Geometry'
import {fGet} from '../../../Utils/Utils'
import {useChartData} from '../../App/WithChartData'
import {useSimulation} from '../../App/WithSimulation'
import {GlidePathInput} from '../../Common/Inputs/GlidePathInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {ChartPanelType} from '../ChartPanel/ChartPanelType'
import {usePlanContent} from '../Plan'
import {AssetAllocationChart} from './Helpers/AssetAllocationChart'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

type Props = ParamsInputBodyPassThruProps & {
  chartType: ChartPanelType | 'sharpe-ratio'
  setChartType: (type: ChartPanelType | 'sharpe-ratio') => void
}
export const ParamsInputStrategy = React.memo((props: Props) => {
  const {params, setParams} = useSimulation()
  const content = usePlanContent()['strategy']

  const handleStrategy = (strategy: TPAWParams['strategy']) => {
    setParams(p => {
      const clone = _.cloneDeep(p)
      clone.strategy = strategy
      return clone
    })
  }

  return (
    <ParamsInputBody {...props} headingMarginLeft="reduced">
      <div className="">
        <div
          className="px-2"
          style={{
            ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
          }}
        >
          <Contentful.RichText body={content.intro.fields.body} p="p-base" />
        </div>

        <div className="mt-8">
          <div
            className=" outline-none params-card"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <button
              className=" text-left"
              onClick={() => handleStrategy('TPAW')}
            >
              <div className={`cursor-pointer `}>
                <h2 className=" font-bold text-lg">
                  <FontAwesomeIcon
                    className="mr-2"
                    icon={
                      params.strategy === 'TPAW'
                        ? faCircleSelected
                        : faCircleRegular
                    }
                  />{' '}
                  Total portfolio approach
                </h2>
                <div className="mt-2">
                  <Contentful.RichText
                    body={content.tpawIntro.fields.body}
                    p="p-base"
                  />
                </div>
              </div>
            </button>
          </div>

          <div
            className="params-card outline-none mt-8"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <button
              className={`text-left  `}
              onClick={() => handleStrategy('SPAW')}
            >
              <h2 className=" font-bold text-lg">
                <FontAwesomeIcon
                  className="mr-2"
                  icon={
                    params.strategy === 'SPAW'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />{' '}
                Savings portfolio approach
              </h2>
              <div className="mt-2">
                <Contentful.RichText
                  body={content.spawIntro.fields.body}
                  p="p-base"
                />
              </div>
            </button>
            {params.strategy === 'SPAW' && (
              <div className="mt-8">
                <h2 className="font-bold text-l">
                  Asset Allocation on the Savings Portfolio
                </h2>

                <GlidePathInput
                  className="mt-4 border border-gray-300 p-2 rounded-lg"
                  value={params.targetAllocation.regularPortfolio.forSPAW}
                  onChange={x =>
                    setParams(p => {
                      const clone = _.cloneDeep(p)
                      clone.targetAllocation.regularPortfolio.forSPAW = x
                      return clone
                    })
                  }
                />
                <h2 className="mt-6">Graph of this asset allocation:</h2>
                <AssetAllocationChart className="mt-4 " />
              </div>
            )}
          </div>
        </div>

        <_ComparisonCard className="mt-8 params-card " props={props} />
      </div>
    </ParamsInputBody>
  )
})

const _ComparisonCard = React.memo(
  ({className = '', props}: {className?: string; props: Props}) => {
    const {chartType, setChartType} = props
    const {setCompareSharpeRatio} = useSimulation()

    const content = usePlanContent()['strategy']
    const chartData = useChartData().sharpeRatio

    // Remember the last non sharpe ratio chart type.
    const [lastNonShapeRatioChartType, setLastNonSharpeRatioChartType] =
      useState(chartType === 'sharpe-ratio' ? null : chartType)

    console.log(lastNonShapeRatioChartType)

    useEffect(() => {
      if (chartType !== 'sharpe-ratio')
        setLastNonSharpeRatioChartType(chartType)
    }, [chartType])

    // Flag for showing sharpe ratio.
    const [showSharpeRatio, setShowShapeRatio] = useState(false)

    // Sync byStrategy to flag.
    useEffect(() => {
      setCompareSharpeRatio(showSharpeRatio)
    }, [setCompareSharpeRatio, showSharpeRatio])

    // Sync chart type to flag and byStrategy.
    useEffect(() => {
      if (showSharpeRatio) {
        if (chartData) setChartType('sharpe-ratio')
      } else {
        if (chartType === 'sharpe-ratio' && lastNonShapeRatioChartType)
          setChartType(lastNonShapeRatioChartType)
      }
    }, [
      chartData,
      chartType,
      lastNonShapeRatioChartType,
      setChartType,
      showSharpeRatio,
    ])

    // If chart type changes from outside.
    useEffect(() => {
      if (chartType !== 'sharpe-ratio') {
        setShowShapeRatio(false)
      }
    }, [chartType])

    // Cleanup.
    const cleanupRef = useRef<() => void>()
    cleanupRef.current = () => {
      if (chartType === 'sharpe-ratio' && lastNonShapeRatioChartType) {
        setChartType(lastNonShapeRatioChartType)
      }
      setCompareSharpeRatio(false)
    }
    useEffect(() => () => fGet(cleanupRef.current)(), [])

    return (
      <div
        className={`${className}`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">Compare reward/risk ratio</h2>

        <div className="mt-2">
          <Contentful.RichText
            body={content.sharpeRatioIntro.fields.body}
            p="p-base"
          />
        </div>
        <div className="inline-flex gap-x-2 items-center py-1 mt-2">
          <button
            className={``}
            onClick={() => setShowShapeRatio(!showSharpeRatio)}
          >
            Show reward/risk ratio
          </button>
          <ToggleSwitch
            enabled={showSharpeRatio}
            setEnabled={setShowShapeRatio}
          />
        </div>
      </div>
    )
  }
)
