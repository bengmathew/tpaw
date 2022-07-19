import {faCaretDown, faCaretRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {useState} from 'react'
import {getDefaultParams} from '../../../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../../../TPAWSimulator/TPAWParams'
import {TPAWParamsExt} from '../../../../TPAWSimulator/TPAWParamsExt'
import {Padding, paddingCSSStyleHorz} from '../../../../Utils/Geometry'
import {noCase} from '../../../../Utils/Utils'
import {Footer} from '../../../App/Footer'
import {useSimulation} from '../../../App/WithSimulation'
import {Config} from '../../../Config'
import {analyzeYearsInParams} from '../Helpers/AnalyzeYearsInParams'
import {ParamsInputType} from '../Helpers/ParamsInputType'
import {Reset} from '../Reset'
import {Share} from '../Share'
import {ParamsInputSummaryButton} from './ParamsInputSummaryButton'
import {ParamsInputSummaryStrategy} from './ParamsInputSummaryStrategy'

export const ParamsInputSummary = React.memo(
  ({
    layout,
    state,
    setState,
    cardPadding,
  }: {
    layout: 'mobile' | 'desktop' | 'laptop'
    state: ParamsInputType | 'summary'
    setState: (state: ParamsInputType) => void
    cardPadding: Padding
  }) => {
    const {params, paramsExt} = useSimulation()
    const {asYFN, withdrawalStartYear} = paramsExt
    const isRetired = asYFN(withdrawalStartYear) <= 0

    const [showAdvanced, setShowAdvanced] = useState(false)
    const advancedModifiedCount = _advancedInputs.filter(x =>
      _isModified(x, params)
    ).length

    return (
      <div
        className={` grid w-full`} // This is needed if we start in exited state.
        style={{grid: '1fr auto/1fr'}}
      >
        <div
          className={`flex flex-col items-start mb-16  ${
            layout === 'desktop' ? 'max-w-[600px]' : ''
          }`}
        >
          <div className="mb-4 w-full">
            <div
              className="flex justify-end items-center my-1"
              style={{paddingLeft: `${cardPadding.left}px`}}
            >
              <div className={`flex gap-x-4 `}>
                <Reset />
                <Share />
              </div>
            </div>
            <ParamsInputSummaryStrategy className="" setState={setState} />
          </div>
          <div
            className={`flex flex-col gap-y-12 sm:gap-y-16 relative z-0 w-full`}
          >
            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={{...paddingCSSStyleHorz(cardPadding)}}
              >
                Basic Inputs
              </h2>
              <div className="flex flex-col gap-y-4 ">
                <ParamsInputSummaryButton
                  type="age-and-retirement"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                />
                <ParamsInputSummaryButton
                  type="current-portfolio-balance"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                />
                {!isRetired && (
                  <ParamsInputSummaryButton
                    type="future-savings"
                    setState={setState}
                    state={state}
                    warn={!_paramsOk(paramsExt, 'future-savings')}
                    padding={cardPadding}
                  />
                )}
                <ParamsInputSummaryButton
                  type="income-during-retirement"
                  setState={setState}
                  state={state}
                  warn={!_paramsOk(paramsExt, 'income-during-retirement')}
                  padding={cardPadding}
                />
              </div>
            </div>

            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={{...paddingCSSStyleHorz(cardPadding)}}
              >
                Spending Goals
              </h2>
              <div className="flex flex-col gap-y-4">
                <ParamsInputSummaryButton
                  type="extra-spending"
                  setState={setState}
                  state={state}
                  warn={!_paramsOk(paramsExt, 'extra-spending')}
                  padding={cardPadding}
                />
                <ParamsInputSummaryButton
                  type="legacy"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                  strategies={['TPAW', 'SPAW']}
                />
              </div>
            </div>

            <div className="">
              <h2
                className="text-[18px] sm:text-xl font-bold mb-4"
                style={{...paddingCSSStyleHorz(cardPadding)}}
              >
                Risk and Time Preference
              </h2>
              <div className="flex flex-col gap-y-4">
                <ParamsInputSummaryButton
                  type="stock-allocation"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                />
                <ParamsInputSummaryButton
                  type="spending-tilt"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                  strategies={['TPAW', 'SPAW']}
                />
                <ParamsInputSummaryButton
                  type="spending-ceiling-and-floor"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                  strategies={['TPAW', 'SPAW']}
                />
                {!Config.client.production && (
                  <ParamsInputSummaryButton
                    type="lmp"
                    setState={setState}
                    state={state}
                    padding={cardPadding}
                    strategies={['TPAW', 'SPAW']}
                  />
                )}
                <ParamsInputSummaryButton
                  type="withdrawal"
                  setState={setState}
                  state={state}
                  padding={cardPadding}
                  strategies={['SWR']}
                />
              </div>
            </div>

            <div className="">
              <button
                className=""
                style={{...paddingCSSStyleHorz(cardPadding)}}
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <div className="text-[20px] sm:text-xl2 font-bold text-left">
                  Advanced
                  <FontAwesomeIcon
                    className="ml-2"
                    icon={showAdvanced ? faCaretDown : faCaretRight}
                  />
                </div>
                {!showAdvanced && (
                  <h2 className="text-left">
                    {advancedModifiedCount === 0
                      ? 'None'
                      : `${advancedModifiedCount} modified`}
                  </h2>
                )}
              </button>
              {showAdvanced && (
                <div className="flex flex-col gap-y-4 mt-4">
                  <ParamsInputSummaryButton
                    type="expected-returns"
                    setState={setState}
                    state={state}
                    padding={cardPadding}
                    flagAsModified={_isModified('expected-returns', params)}
                  />
                  <ParamsInputSummaryButton
                    type="inflation"
                    setState={setState}
                    state={state}
                    padding={cardPadding}
                    flagAsModified={_isModified('inflation', params)}
                  />
                  <ParamsInputSummaryButton
                    type="compare-strategies"
                    setState={setState}
                    state={state}
                    padding={cardPadding}
                    flagAsModified={_isModified('compare-strategies', params)}
                  />
                  <ParamsInputSummaryButton
                    type="simulation"
                    setState={setState}
                    state={state}
                    padding={cardPadding}
                    flagAsModified={_isModified('simulation', params)}
                  />
                  {!Config.client.production && (
                    <ParamsInputSummaryButton
                      type="dev"
                      setState={setState}
                      state={state}
                      padding={cardPadding}
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
        {layout !== 'laptop' && <Footer />}
      </div>
    )
  }
)

type _Type =
  | 'future-savings'
  | 'income-during-retirement'
  | 'extra-spending'
  | 'compare-strategies'
export const _paramsOk = (paramsExt: TPAWParamsExt, type: _Type) => {
  const {valueForYearRange, glidePath} = analyzeYearsInParams(paramsExt)
  return (
    valueForYearRange
      .filter(x => x.section === type)
      .every(x => x.boundsCheck.start === 'ok' && x.boundsCheck.end === 'ok') &&
    glidePath
      .filter(x => x.section === type)
      .every(x => x.analyzed.every(x => x.issue === 'none'))
  )
}

const _advancedInputs = [
  'expected-returns',
  'inflation',
  'compare-strategies',
  'simulation',
] as const
type _AdvancedParamInputType = typeof _advancedInputs[number]
const _isModified = (type: _AdvancedParamInputType, params: TPAWParams) => {
  const def = getDefaultParams()
  switch (type) {
    case 'expected-returns':
      return params.returns.expected.type !== 'suggested'
    case 'inflation':
      return params.inflation.type !== 'suggested'
    case 'compare-strategies':
      return params.strategy !== def.strategy
    case 'simulation':
      return params.sampling !== def.sampling
    default:
      noCase(type)
  }
}
