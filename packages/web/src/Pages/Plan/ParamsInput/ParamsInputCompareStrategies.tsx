import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {defaultSWRWithdrawalRate} from '../../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS, paddingCSSStyleHorz} from '../../../Utils/Geometry'
import {fGet} from '../../../Utils/Utils'
import {useChartData} from '../../App/WithChartData'
import {useSimulation} from '../../App/WithSimulation'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {Config} from '../../Config'
import {ChartPanelType} from '../ChartPanel/ChartPanelType'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

type Props = ParamsInputBodyPassThruProps & {
  chartType: ChartPanelType | 'sharpe-ratio'
  setChartType: (type: ChartPanelType | 'sharpe-ratio') => void
}
export const ParamsInputCompareStrategies = React.memo((props: Props) => {
  const {params, setParams, paramsExt} = useSimulation()

  const content = usePlanContent()['compare-strategies']

  const handleStrategy = (strategy: TPAWParams['strategy']) => {
    setParams(params => {
      if (params.strategy === strategy) return params
      const clone = _.cloneDeep(params)
      clone.strategy = strategy
      if (strategy === 'SWR') {
        clone.swrWithdrawal = {
          type: 'asPercent',
          percent: defaultSWRWithdrawalRate(paramsExt.numRetirementYears),
        }
      }
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
          <Contentful.RichText
            body={content.intro[params.strategy]}
            p="p-base"
          />
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
                  TPAW –{' '}
                  <span className="">
                    Total Portfolio Allocation and Withdrawal
                  </span>
                </h2>
                {/* <h2 className=" mt-2 font-semibold ">
                
              </h2> */}

                <div className="mt-2">
                  <Contentful.RichText
                    body={content.tpawIntro[params.strategy]}
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
                SPAW –{' '}
                <span className="">
                  Savings Portfolio Allocation and Withdrawal
                </span>
              </h2>
              {/* <h2 className=" font-semibold mt-2 ">
                Savings Portfolio Allocation and Withdrawal
              </h2> */}
              <div className="mt-2">
                <Contentful.RichText
                  body={content.spawIntro[params.strategy]}
                  p="p-base"
                />
              </div>
            </button>
          </div>
        </div>

        {!Config.client.production && (
          <div
            className="params-card outline-none mt-8"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <button
              className={`text-left  `}
              onClick={() => handleStrategy('SWR')}
            >
              <h2 className=" font-bold text-lg">
                <FontAwesomeIcon
                  className="mr-2"
                  icon={
                    params.strategy === 'SWR'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />{' '}
                SWR – <span className="">Safe Withdrawal Rate</span>
              </h2>
              {/* <h2 className=" font-semibold mt-2 ">
    Savings Portfolio Allocation and Withdrawal
  </h2> */}
              <div className="mt-2">
                <Contentful.RichText
                  body={content.spawIntro[params.strategy]}
                  p="p-base"
                />
              </div>
            </button>
          </div>
        )}

        <_ComparisonCard className="mt-8 params-card " props={props} />
      </div>
    </ParamsInputBody>
  )
})

const _ComparisonCard = React.memo(
  ({className = '', props}: {className?: string; props: Props}) => {
    const {chartType, setChartType} = props
    const {setCompareSharpeRatio, params} = useSimulation()

    const content = usePlanContent()['compare-strategies']
    const chartData = useChartData().sharpeRatio

    // Remember the last non sharpe ratio chart type.
    const [lastNonShapeRatioChartType, setLastNonSharpeRatioChartType] =
      useState(chartType === 'sharpe-ratio' ? null : chartType)

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
            body={content.sharpeRatioIntro[params.strategy]}
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
