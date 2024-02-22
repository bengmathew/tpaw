import { GlidePath } from '@tpaw/common'
import React from 'react'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { GlidePathInput } from '../../../../Common/Inputs/GlidePathInput'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { AssetAllocationChart } from '../Helpers/AssetAllocationChart'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'

export const PlanInputRiskSPAWAndSWRStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams, defaultPlanParams } = useSimulation()
    const handleChange = (x: GlidePath) =>
      updatePlanParams('setSPAWAndSWRAllocation', x)

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">Stock Allocation</h2>
        <GlidePathInput
          className="mt-4 border border-gray-300 p-2 rounded-lg"
          value={planParams.risk.spawAndSWR.allocation}
          onChange={handleChange}
        />
        <h2 className="mt-6">Graph of this asset allocation:</h2>
        <AssetAllocationChart
          className="mt-4 "
          type="asset-allocation-savings-portfolio"
        />
        <button
          className="mt-6 underline"
          onClick={() => handleChange(defaultPlanParams.risk.spawAndSWR.allocation)}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)
