import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {GlidePathInput} from '../../Common/Inputs/GlidePathInput'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {AssetAllocationChart} from './Helpers/AssetAllocationChart'
import {ParamsInputStrategyConditionCard} from './Helpers/ParamsInputStrategyConditionCard'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputStockAllocation = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const content = usePlanContent()['stock-allocation']

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <_TotalStockAllocationCard className="" props={props} />
          <_SavingsStockAllocationCard className="" props={props} />
        </div>
      </ParamsInputBody>
    )
  }
)

const _TotalStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['stock-allocation']

    return (
      <ParamsInputStrategyConditionCard
        className={`${className} params-card`}
        props={props}
        tpaw
        spaw={false}
        swr={false}
      >
        <h2 className="font-bold text-lg">
          Stock Allocation on Total Portfolio
        </h2>
        <div className="mt-2">
          <Contentful.RichText body={content.intro[params.strategy]} p="p-base" />
        </div>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          pointers={[
            {
              value: params.targetAllocation.regularPortfolio.forTPAW.stocks,
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
        {/* {params.strategy === 'TPAW' && (
          <div className="p-base mt-2">
            <span className="bg-gray-300 px-2 rounded-lg ">Note</span> The stock
            allocation you set here is on your{' '}
            <span className="">total portfolio</span>. To view the resulting
            asset allocation on your savings portfolio, select{' '}
            {`"${chartPanelLabel(params, 'glide-path', 'short').label.join(
              ' '
            )}"`}{' '}
            from the drop down menu of the graph.
          </div>
        )} */}
      </ParamsInputStrategyConditionCard>
    )
  }
)

export const _SavingsStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()

    const content = usePlanContent()['stock-allocation']
    return (
      <ParamsInputStrategyConditionCard
        className={`${className} params-card ${
          params.strategy === 'TPAW' ? 'lighten' : ''
        }`}
        props={props}
        tpaw={false}
        spaw
        swr
      >
        <h2 className="font-bold text-lg">
          Stock Allocation on Savings Portfolio
        </h2>
        <div className="mt-2">
          <Contentful.RichText body={content.intro[params.strategy]} p="p-base" />
        </div>
        <GlidePathInput
          className="mt-4 border border-gray-300 p-2 rounded-lg"
          value={params.targetAllocation.regularPortfolio.forSPAWAndSWR}
          onChange={x =>
            setParams(p => {
              const clone = _.cloneDeep(p)
              clone.targetAllocation.regularPortfolio.forSPAWAndSWR = x
              return clone
            })
          }
        />
        <h2 className="mt-6">Graph of this asset allocation:</h2>
        <AssetAllocationChart className="mt-4 " />
      </ParamsInputStrategyConditionCard>
    )
  }
)
