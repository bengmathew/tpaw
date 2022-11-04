import { faCaretDown, faCheck, faSave } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { resolveTPAWRiskPreset, TPAWRiskLevel } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useState } from 'react'
import { getNumYears } from '../../../TPAWSimulator/PlanParamsExt'
import { riskLevelLabel } from '../../../TPAWSimulator/RiskLevelLabel'
import { assert, fGet } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { BasicMenu } from '../../Common/Modal/BasicMenu'

export const PlanSummaryRiskCopyFromAPreset = React.memo(
  ({ className = '' }: { className?: string }) => {
    const { params, setParams } = useSimulation()

    const handleClick = (
      closeMenu: () => void,
      preset: TPAWRiskLevel | 'saved',
    ) => {
      closeMenu()
      setParams((params) => {
        const clone = _.cloneDeep(params)
        assert(!clone.risk.useTPAWPreset)
        const tpawRisk = _.cloneDeep(
          preset === 'saved'
            ? fGet(clone.risk.savedTPAWPreset)
            : resolveTPAWRiskPreset(
                { ...clone.risk, useTPAWPreset: true, tpawPreset: preset },
                getNumYears(clone),
              ),
        )
        clone.risk = { ...clone.risk, ...tpawRisk }
        return clone
      })
    }

    const [markAsSaved, setMarkAsSaved] = useState(false)
    useEffect(() => {
      const timeout = window.setTimeout(() => setMarkAsSaved(false), 750)
      return () => window.clearTimeout(timeout)
    }, [markAsSaved])

    return (
      <BasicMenu align="left">
        <div className={`${className} flex items-center gap-x-2`}>
          Copy from a Preset
          <FontAwesomeIcon className="text-lg" icon={faCaretDown} />
        </div>
        {(closeMenu) => (
          <div className="">
            <div className="flex flex-col mt-4   ">
              <h2 className="font-bold mx-4 text-xl">Presets</h2>
              <button
                className="text-left py-2 px-4 font-medium"
                onClick={() => handleClick(closeMenu, 'riskLevel-1')}
              >
                {riskLevelLabel('riskLevel-1')}
              </button>
              <button
                className="text-left py-2 px-4 font-medium "
                onClick={() => handleClick(closeMenu, 'riskLevel-2')}
              >
                {riskLevelLabel('riskLevel-2')}
              </button>
              <button
                className="text-left py-2 px-4 font-medium "
                onClick={() => handleClick(closeMenu, 'riskLevel-3')}
              >
                {riskLevelLabel('riskLevel-3')}
              </button>
              <button
                className="text-left py-2 px-4 font-medium "
                onClick={() => handleClick(closeMenu, 'riskLevel-4')}
              >
                {riskLevelLabel('riskLevel-4')}
              </button>
            </div>
            <div className=" px-4 text-left font-medium mt-4">
              <h2 className="font-bold text-lg">Custom</h2>
              <button
                className=" disabled:lighten-2 py-2"
                disabled={params.risk.savedTPAWPreset === null}
                onClick={() => handleClick(closeMenu, 'saved')}
              >
                {markAsSaved ? (
                  <span>
                    <FontAwesomeIcon icon={faCheck} /> Saved
                  </span>
                ) : (
                  'Last Saved Risk Profile'
                )}
              </button>
              <div className="flex justify-end mt-1">
                <button
                  className="flex items-center gap-x-2 py-2 mb-2  btn-sm rounded-full bg-gray-300 "
                  onClick={() => {
                    setMarkAsSaved(true)
                    setParams((params) => {
                      assert(!params.risk.useTPAWPreset)
                      const clone = _.cloneDeep(params)
                      clone.risk.savedTPAWPreset = _.cloneDeep({
                        tpaw: params.risk.tpaw,
                        tpawAndSPAW: params.risk.tpawAndSPAW,
                      })
                      return clone
                    })
                  }}
                >
                  <FontAwesomeIcon icon={faSave} />
                  Save Now
                </button>
              </div>
            </div>
          </div>
        )}
      </BasicMenu>
    )
  },
)
