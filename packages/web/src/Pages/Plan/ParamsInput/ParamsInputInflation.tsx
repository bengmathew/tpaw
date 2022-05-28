import React from 'react'
import { getDefaultParams } from '../../../TPAWSimulator/DefaultParams'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS } from '../../../Utils/Geometry'
import { preciseRange } from '../../../Utils/PreciseRange'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { usePlanContent } from '../Plan'
import { ParamsInputBody, ParamsInputBodyPassThruProps } from './ParamsInputBody'

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
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content.inflation.intro.fields.body}
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
              onChange={([inflation]) =>
                setParams(params => ({...params, inflation}))
              }
            />
          </div>
          <button
            className="mt-4 underline"
            onClick={() =>
              setParams(p => ({
                ...p,
                inflation: getDefaultParams().inflation,
              }))
            }
          >
            Reset to Default
          </button>
        </div>
      </ParamsInputBody>
    )
  }
)
