import _ from 'lodash'
import React from 'react'
import {getDefaultParams} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputInflation = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
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

    const handleChange = (inflation: number) => {
      setParams(params => {
        const clone = _.cloneDeep(params)
        clone.inflation = inflation
        return clone
      })
    }
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content.inflation.intro[params.strategy]}
            p="col-span-2 mb-2 p-base"
          />
          <div
            className="grid mt-4 items-center"
            style={{grid: 'auto / auto 1fr'}}
          >
            <h2 className="">Inflation</h2>
            <SliderInput
              {...sliderProps}
              pointers={[{value: params.inflation, type: 'normal'}]}
              onChange={([inflation]) => handleChange(inflation)}
            />
          </div>
          <button
            className="mt-4 underline"
            onClick={() => handleChange(getDefaultParams().inflation)}
          >
            Reset to Default
          </button>
        </div>
      </ParamsInputBody>
    )
  }
)
