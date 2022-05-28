import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSSStyle, paddingCSSStyleHorz} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {chartPanelLabel} from '../ChartPanel/ChartPanelLabel'
import {usePlanContent} from '../Plan'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputRiskAndTimePreference = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['risk-and-time-preference']
    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <div
            className=""
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
            }}
          >
            <Contentful.RichText
              body={content.intro.fields.body}
              p="mb-2 p-base"
            />
          </div>
          <div
            className="params-card mt-10"
            style={{...paddingCSSStyle(props.sizing.cardPadding)}}
          >
            <h2 className="font-bold text-lg">Stock Allocation</h2>
            <div className="mt-2">
              <Contentful.RichText
                body={content.stockAllocationIntro.fields.body}
                p="p-base"
              />
            </div>
            {params.strategy === 'SPAW' && (
              <div className="p-base mt-2">
                <span className="bg-gray-300 px-2 rounded-lg ">Note</span>{' '}
                {`You have selected the savings portfolio approach in the "${paramsInputLabel(
                  'strategy'
                )}" section. This means that the stock allocation you set here is ignored.`}
              </div>
            )}
            {/* <AssetAllocationChart className="" /> */}
            <SliderInput
              className={`-mx-3 mt-2
              ${
                params.strategy !== 'TPAW'
                  ? 'lighten-2 pointer-events-none'
                  : ''
              }`}
              height={60}
              pointers={[
                {
                  value:
                    params.targetAllocation.regularPortfolio.forTPAW.stocks,
                  type: 'normal',
                },
              ]}
              onChange={([value]) =>
                setParams(params => {
                  const p = _.cloneDeep(params)
                  p.targetAllocation.regularPortfolio.forTPAW.stocks = value
                  return p
                })
              }
              formatValue={formatPercentage(0)}
              domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
                value: value,
                tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
              }))}
            />
            {params.strategy === 'TPAW' && (
              <div className="p-base mt-2">
                <span className="bg-gray-300 px-2 rounded-lg ">Note</span> The
                stock allocation you set here is on your{' '}
                <span className="">total portfolio</span>. To view the resulting
                asset allocation on your savings portfolio, select {' '}
                {`"${chartPanelLabel(params, 'glide-path', 'short').label.join(
                  ' '
                )}"`}{' '}
                from the drop down menu of the graph.
              </div>
            )}
          </div>

          <div
            className="params-card mt-10"
            style={{...paddingCSSStyle(props.sizing.cardPadding)}}
          >
            <h2 className="font-bold text-lg">Spending Tilt</h2>
            <div className="mt-2">
              <Contentful.RichText
                body={content.spendingTiltIntro.fields.body}
                p="p-base"
              />
            </div>
            <SliderInput
              className="-mx-3 mt-2"
              height={60}
              pointers={[
                {value: params.scheduledWithdrawalGrowthRate, type: 'normal'},
              ]}
              onChange={([value]) =>
                setParams(params => ({
                  ...params,
                  scheduledWithdrawalGrowthRate: value,
                }))
              }
              formatValue={formatPercentage(1)}
              domain={preciseRange(-0.03, 0.03, 0.001, 3).map((value, i) => ({
                value,
                tick: i % 10 === 0 ? 'large' : i % 1 === 0 ? 'small' : 'none',
              }))}
            />
          </div>
        </div>
      </ParamsInputBody>
    )
  }
)
