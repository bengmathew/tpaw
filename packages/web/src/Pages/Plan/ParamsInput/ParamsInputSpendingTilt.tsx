import _ from 'lodash'
import React from 'react'
import {getDefaultParams} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputSpendingTilt = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <_SpendingTilt className="" props={props} />
      </ParamsInputBody>
    )
  }
)

const _SpendingTilt = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['spending-tilt']

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <div className="">
          <Contentful.RichText
            body={content.intro[params.strategy]}
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
        <button
          className="mt-4 underline"
          onClick={() => {
            setParams(params => {
              const p = _.cloneDeep(params)
              p.scheduledWithdrawalGrowthRate =
                getDefaultParams().scheduledWithdrawalGrowthRate
              return p
            })
          }}
        >
          Reset to Default
        </button>
      </div>
    )
  }
)
