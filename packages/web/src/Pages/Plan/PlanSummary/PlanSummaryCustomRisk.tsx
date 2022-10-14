import _ from 'lodash'
import React, {useState} from 'react'
import {resolveTPAWRiskPreset} from '../../../TPAWSimulator/DefaultParams'
import {TPAWParams, TPAWRisk} from '../../../TPAWSimulator/TPAWParams'
import {useSimulation} from '../../App/WithSimulation'
import {ConfirmAlert} from '../../Common/Modal/ConfirmAlert'

export const PlanSummaryCustomRisk = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, setParams} = useSimulation()
    const [showPopup, setShowPopup] = useState(false)

    if(params.strategy !== 'TPAW') return <></>
    return (
      <>
        <button
          className={`${className} text-sm font-normal underline`}
          onClick={() => {
            setShowPopup(true)
          }}
        >
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
              setParams(params => {
                const clone = _.cloneDeep(params)
                if (params.risk.useTPAWPreset) {
                  clone.risk = resolveTPAWRiskPreset(clone.risk)
                } else {
                  clone.risk.customTPAWPreset = _customIfNeeded(params.risk)
                  if (clone.risk.customTPAWPreset) {
                    clone.risk.tpawPreset = 'custom'
                  }
                  clone.risk.useTPAWPreset = true
                }
                return clone
              })
            }}
          >
            {params.risk.useTPAWPreset
              ? `Custom mode gives you more control over your risk strategy.`
              : !_customIfNeeded(params.risk)
              ? `Are you sure you want to switch back to preset mode?`
              : params.risk.customTPAWPreset
              ? `Any changes you made to your risk strategy will be saved to your custom preset.`
              : `Any changes you made to your risk strategy will be saved as a custom preset.`}
          </ConfirmAlert>
        )}
      </>
    )
  }
)

const _customIfNeeded = (
  risk: Exclude<TPAWParams['risk'], {useTPAWPreset: true}>
): TPAWRisk | null => {
  const fromPreset = _.pick(
    resolveTPAWRiskPreset({...risk, useTPAWPreset: true}),
    ['tpaw', 'tpawAndSPAW']
  )
  const custom = {
    tpaw: _.cloneDeep(risk.tpaw),
    tpawAndSPAW: _.cloneDeep(risk.tpawAndSPAW),
  }
  return risk.customTPAWPreset || !_.isEqual(custom, fromPreset) ? custom : null
}
