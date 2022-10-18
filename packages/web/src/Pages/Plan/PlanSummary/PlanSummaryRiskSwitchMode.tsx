import _ from 'lodash'
import React, {useState} from 'react'
import {resolveTPAWRiskPreset} from '../../../TPAWSimulator/DefaultParams'
import {useSimulation} from '../../App/WithSimulation'
import {ConfirmAlert} from '../../Common/Modal/ConfirmAlert'

export const PlanSummaryRiskSwitchMode = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, paramsExt, setParams} = useSimulation()
    const [showPopup, setShowPopup] = useState(false)

    if (params.strategy !== 'TPAW') return <></>
    return (
      <div
        className={`${className} flex flex-col items-end text-sm font-normal`}
      >
        <button className="underline" onClick={() => setShowPopup(true)}>
          {params.risk.useTPAWPreset
            ? `Switch to custom mode`
            : `Switch to preset mode`}
        </button>
        {showPopup && (
          <ConfirmAlert
            title={null}
            confirmText={
              params.risk.useTPAWPreset
                ? 'Switch to Custom Mode'
                : `Switch to Preset Mode`
            }
            onCancel={() => setShowPopup(false)}
            onConfirm={() => {
              setShowPopup(false)
              setParams(() => {
                const clone = _.cloneDeep(params)
                if (clone.risk.useTPAWPreset) {
                  const tpawRisk = _.cloneDeep(
                    clone.risk.customTPAWPreset ??
                      resolveTPAWRiskPreset(clone.risk, paramsExt.numYears)
                  )
                  clone.risk = {
                    useTPAWPreset: false,
                    tpawPreset: clone.risk.tpawPreset,
                    customTPAWPreset: clone.risk.customTPAWPreset,
                    savedTPAWPreset: clone.risk.savedTPAWPreset,
                    tpaw: tpawRisk.tpaw,
                    tpawAndSPAW: tpawRisk.tpawAndSPAW,
                    spawAndSWR: clone.risk.spawAndSWR,
                    swr: clone.risk.swr,
                  }
                } else {
                  clone.risk = {
                    useTPAWPreset: true,
                    tpawPreset: clone.risk.tpawPreset,
                    customTPAWPreset: _.cloneDeep({
                      tpaw: clone.risk.tpaw,
                      tpawAndSPAW: clone.risk.tpawAndSPAW,
                    }),
                    savedTPAWPreset: clone.risk.savedTPAWPreset,
                    spawAndSWR: clone.risk.spawAndSWR,
                    swr: clone.risk.swr,
                  }
                }
                return clone
              })
            }}
          >
            {params.risk.useTPAWPreset
              ? `Are you sure you want to switch to custom mode?`
              : `Are you sure you want to switch to preset mode?`}
          </ConfirmAlert>
        )}
      </div>
    )
  }
)
