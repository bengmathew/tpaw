import _ from 'lodash'
import React from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {formatPercentage} from '../../Utils/FormatPercentage'
import {preciseRange} from '../../Utils/PreciseRange'
import {StateObj} from '../../Utils/UseStateObj'
import {SliderInput} from '../Common/Inputs/SliderInput/SliderInput'

export const MainControls = React.memo(
  ({
    params: paramsObj,
    className = '',
  }: {
    params: StateObj<TPAWParams>
    className?: string
  }) => {
    const {value: params, set: setParams} = paramsObj
    return (
      <div
        className={`${className} grid items-center md:gap-x-5 lg:gap-x-10  `}
        style={{grid: ' auto / 1fr 1fr '}}
      >
        <h2 className="font-medium text-center sm:text-left sm:pl-[20px] mt-2 whitespace-nowrap">
          Stock Allocation
        </h2>
        <h2 className="font-medium text-center sm:text-left sm:pl-[20px] mt-2 whitespace-nowrap">
          Spending Growth
        </h2>
        <SliderInput
          className=""
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
        <SliderInput
          className=""
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
    )
  }
)
