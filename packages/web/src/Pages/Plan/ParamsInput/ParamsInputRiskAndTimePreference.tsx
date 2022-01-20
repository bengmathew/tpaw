import _ from 'lodash'
import React from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { preciseRange } from '../../../Utils/PreciseRange'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { usePlanContent } from '../Plan'

export const ParamsInputRiskAndTimePreference = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()
    return (
      <div className="">
        <Contentful.RichText
          body={content.riskAndTimePreference.intro.fields.body}
          p="mb-2"
        />
        <div
          className="grid my-2 items-center"
          style={{grid: 'auto / auto 1fr'}}
        >
          <h2 className=" whitespace-nowrap">Stock Allocation</h2>
          <SliderInput
            className="-mx-3"
            height={60}
            pointers={[
              {
                value: params.targetAllocation.regularPortfolio.stocks,
                type: 'normal',
              },
            ]}
            onChange={([value]) =>
              setParams(params => {
                const p = _.cloneDeep(params)
                p.targetAllocation.regularPortfolio.stocks = value
                return p
              })
            }
            formatValue={formatPercentage(0)}
            domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
              value: value,
              tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
            }))}
          />

          <h2 className=" whitespace-nowrap">Spending Tilt</h2>
          <SliderInput
            className="-mx-3"
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
    )
  }
)
