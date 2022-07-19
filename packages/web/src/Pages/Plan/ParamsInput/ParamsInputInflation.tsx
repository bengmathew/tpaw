import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'

import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {processInflation} from '../../../TPAWSimulator/TPAWParamsProcessed'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSS} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputInflation = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <_InflationCard className="" props={props} />
      </ParamsInputBody>
    )
  }
)

export const _InflationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()
    const sliderProps = {
      className: '',
      height: 60,
      formatValue: (x: number) => `${(x * 100).toFixed(1)}%`,
      domain: preciseRange(-0.01, 0.1, 0.001, 3).map((value, i) => ({
        value,
        tick:
          i % 10 === 0
            ? ('large' as const)
            : i % 2 === 0
            ? ('small' as const)
            : ('none' as const),
      })),
    }

    const handleChange = (inflation: TPAWParams['inflation']) => {
      setParams(params => {
        const clone = _.cloneDeep(params)
        clone.inflation = inflation
        return clone
      })
    }
    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <Contentful.RichText
          body={content.inflation.intro[params.strategy]}
          p="col-span-2 mb-2 p-base"
        />

        <button
          className={`${className} flex gap-x-2 mt-4`}
          onClick={() => handleChange({type: 'suggested'})}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.inflation.type === 'suggested'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({type: 'suggested'})}
            </h2>
            <h2 className="text-left text-sm lighten-2">
              {formatPercentage(1)(processInflation({type: 'suggested'}))}
            </h2>
          </div>
        </button>

        <button
          className={`${className} flex gap-x-2 mt-3`}
          onClick={() =>
            handleChange({
              type: 'manual',
              value: processInflation(params.inflation),
            })
          }
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              params.inflation.type === 'manual'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {inflationTypeLabel({type: 'manual'})}
            </h2>
          </div>
        </button>
        {params.inflation.type === 'manual' && (
          <SliderInput
            {...sliderProps}
            pointers={[
              {value: processInflation(params.inflation), type: 'normal'},
            ]}
            onChange={([value]) => handleChange({type: 'manual', value})}
          />
        )}
      </div>
    )
  }
)

export const inflationTypeLabel = ({
  type,
}: {
  type: TPAWParams['inflation']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Suggested'
    case 'manual':
      return 'Manual'
  }
}
