import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup, Switch} from '@headlessui/react'
import _ from 'lodash'
import React, {useMemo, useState} from 'react'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {Contentful} from '../../../Utils/Contentful'
import {fGet, noCase} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyProps} from './ParamsInputBody'

type _Type = 'none' | 'fixedSpending' | 'separateCeilingAndFloor'

export const ParamsInputSpendingCeilingAndFloor = React.memo(
  (props: ParamsInputBodyProps) => {
    const {params, setParams, tpawResult, highlightPercentiles} =
      useSimulation()

    const {asYFN, withdrawalStartYear} = extendTPAWParams(
      tpawResult.args.params.original
    )
    const withdrawalStartAsYFN = asYFN(withdrawalStartYear)
    const content = usePlanContent()

    const [type, setType] = useState<_Type>(
      params.spendingCeiling === null && params.spendingFloor === null
        ? 'none'
        : params.spendingCeiling === params.spendingFloor
        ? 'fixedSpending'
        : 'separateCeilingAndFloor'
    )

    const {minWithdrawal, maxWithdrawal} = useMemo(() => {
      const last = fGet(
        _.last(tpawResult.withdrawals.total.byPercentileByYearsFromNow)
      ).data
      const first = fGet(
        _.first(tpawResult.withdrawals.total.byPercentileByYearsFromNow)
      ).data
      const maxWithdrawal = Math.max(...last.slice(withdrawalStartAsYFN))
      const minWithdrawal = Math.min(...first.slice(withdrawalStartAsYFN))
      return {minWithdrawal, maxWithdrawal}
    }, [tpawResult, withdrawalStartAsYFN])

    const [lastFixedEntry, setLastFixedEntry] = useState(
      params.spendingCeiling === params.spendingFloor
        ? params.spendingCeiling
        : null
    )

    const fixedValueState = useAmountInputState(lastFixedEntry)
    const handleFixedAmount = (amount: number) => {
      setLastFixedEntry(amount)
      fixedValueState.setAmountStr(`${amount}`)
      const p = _.cloneDeep(params)
      p.spendingCeiling = amount
      p.spendingFloor = amount
      setParams(p)
    }

    const [hasCeiling, setHasCeiling] = useState(
      type === 'fixedSpending' ? false : params.spendingCeiling !== null
    )
    const [lastCeilingEntry, setLastCeilingEntry] = useState(
      params.spendingCeiling
    )
    const defaultCeilingAmount = _roundUp(
      minWithdrawal + (maxWithdrawal - minWithdrawal) / 2,
      10000
    )

    const ceilingValueState = useAmountInputState(params.spendingCeiling)
    const handleCeilingAmount = (amount: number) => {
      if (amount === params.spendingCeiling) return
      amount = Math.max(params.spendingFloor ?? 0, amount)
      setLastCeilingEntry(amount)
      ceilingValueState.setAmountStr(`${amount}`)
      const p = _.cloneDeep(params)
      p.spendingCeiling = amount
      setParams(p)
    }

    const [hasFloor, setHasFloor] = useState(
      type === 'fixedSpending' ? false : params.spendingFloor !== null
    )
    const [lastFloorEntry, setLastFloorEntry] = useState(params.spendingFloor)
    const firstWithdrawalOfFirstHighlightPercentile =
      tpawResult.withdrawals.total.byPercentileByYearsFromNow[
        tpawResult.args.percentiles.indexOf(highlightPercentiles[0])
      ].data[withdrawalStartAsYFN]

    const defaultFloorAmount = _roundUp(
      firstWithdrawalOfFirstHighlightPercentile,
      10000
    )
    const floorValueState = useAmountInputState(params.spendingFloor)
    const handleFloorAmount = (amount: number) => {
      if (amount === params.spendingFloor) return
      amount = Math.min(params.spendingCeiling ?? Number.MAX_VALUE, amount)
      setLastFloorEntry(amount)
      floorValueState.setAmountStr(`${amount}`)
      const p = _.cloneDeep(params)
      p.spendingFloor = amount
      setParams(p)
    }

    return (
      <ParamsInputBody {...props}>
        <RadioGroup
          value={`${type}`}
          className=""
          onChange={(type: _Type) => {
            switch (type) {
              case 'fixedSpending':
                handleFixedAmount(lastFixedEntry ?? defaultFloorAmount)
                break
              case 'none': {
                const p = _.cloneDeep(params)
                p.spendingCeiling = null
                p.spendingFloor = null
                setParams(p)
                break
              }
              case 'separateCeilingAndFloor':
                const p = _.cloneDeep(params)
                p.spendingCeiling = hasCeiling ? lastCeilingEntry : null
                p.spendingFloor = hasFloor ? lastFloorEntry : null
                setParams(p)
                break
              default:
                noCase(type)
            }
            setType(type)
          }}
        >
          <RadioGroup.Description>
            <Contentful.RichText
              body={content.spendingCeilingAndFloor.intro.fields.body}
              p="mb-6 p-base"
            />
          </RadioGroup.Description>
          <div className="grid gap-y-6">
            <RadioGroup.Option value="none" className=" outline-none">
              {({checked}) => (
                <div className="">
                  <RadioGroup.Label
                    className={`col-span-2 cursor-pointer ${checked ? '' : ''}`}
                  >
                    <FontAwesomeIcon
                      className="mr-2"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />{' '}
                    No ceiling or floor
                  </RadioGroup.Label>
                </div>
              )}
            </RadioGroup.Option>
            <RadioGroup.Option
              value="separateCeilingAndFloor"
              className=" outline-none"
            >
              {({checked}) => (
                <div className="">
                  <RadioGroup.Label
                    className={`col-span-2 cursor-pointer ${checked ? '' : ''}`}
                  >
                    <FontAwesomeIcon
                      className="mr-2"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />{' '}
                    Separate ceiling and floor
                  </RadioGroup.Label>
                  {checked && (
                    <div
                      className="ml-6 grid gap-x-1 pt-4 items-center"
                      style={{grid: '35px 50px/auto auto 1fr'}}
                    >
                      <Switch.Group>
                        <Switch.Label className="">Ceiling</Switch.Label>
                        <ToggleSwitch
                          className=""
                          enabled={params.spendingCeiling !== null}
                          setEnabled={enabled => {
                            setHasCeiling(enabled)
                            if (!enabled) {
                              const p = _.cloneDeep(params)
                              p.spendingCeiling = null
                              setParams(p)
                            } else {
                              handleCeilingAmount(
                                lastCeilingEntry ?? defaultCeilingAmount
                              )
                            }
                          }}
                        />
                      </Switch.Group>
                      {params.spendingCeiling === null ? (
                        <div className=""></div>
                      ) : (
                        <div className={`flex items-stretch `}>
                          <AmountInput
                            className={`ml-2 w-[90px]`}
                            state={ceilingValueState}
                            onAccept={handleCeilingAmount}
                          />
                          <button
                            className={`flex items-center pl-4 pr-2 `}
                            onClick={() =>
                              handleCeilingAmount(
                                fGet(params.spendingCeiling) + 5000
                              )
                            }
                          >
                            <FontAwesomeIcon
                              className="text-base"
                              icon={faPlus}
                            />
                          </button>
                          <button
                            className={`flex items-center px-2 `}
                            onClick={() =>
                              handleCeilingAmount(
                                Math.max(0, fGet(params.spendingCeiling) - 5000)
                              )
                            }
                          >
                            <FontAwesomeIcon
                              className="text-base"
                              icon={faMinus}
                            />
                          </button>
                        </div>
                      )}
                      <Switch.Group>
                        <Switch.Label className="">Floor</Switch.Label>
                        <ToggleSwitch
                          className=""
                          enabled={params.spendingFloor !== null}
                          setEnabled={enabled => {
                            setHasFloor(enabled)
                            if (!enabled) {
                              const p = _.cloneDeep(params)
                              p.spendingFloor = null
                              setParams(p)
                            } else {
                              handleFloorAmount(
                                lastFloorEntry ?? defaultFloorAmount
                              )
                            }
                          }}
                        />
                      </Switch.Group>{' '}
                      {params.spendingFloor === null ? (
                        <div className=""></div>
                      ) : (
                        <div className={`flex items-stretch `}>
                          <AmountInput
                            className={`ml-2 w-[90px]`}
                            state={floorValueState}
                            onAccept={handleFloorAmount}
                          />
                          <button
                            className={`flex items-center  pl-4 pr-2  `}
                            onClick={() =>
                              handleFloorAmount(
                                fGet(params.spendingFloor) + 5000
                              )
                            }
                          >
                            <FontAwesomeIcon
                              className="text-base"
                              icon={faPlus}
                            />
                          </button>
                          <button
                            className={`flex items-center px-2 `}
                            onClick={() =>
                              handleFloorAmount(
                                Math.max(0, fGet(params.spendingFloor) - 5000)
                              )
                            }
                          >
                            <FontAwesomeIcon
                              className="text-base"
                              icon={faMinus}
                            />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </RadioGroup.Option>
            <RadioGroup.Option value="fixedSpending" className=" outline-none">
              {({checked}) => (
                <div className="">
                  <RadioGroup.Label
                    className={`col-span-2 cursor-pointer ${checked ? '' : ''}`}
                  >
                    <FontAwesomeIcon
                      className="mr-2"
                      icon={checked ? faCircleSelected : faCircleRegular}
                    />{' '}
                    Fixed spending <span className="">(ceiling = floor)</span>
                  </RadioGroup.Label>
                  {checked && (
                    <div
                      className={` mt-4 col-span-2 ml-6 flex items-stretch `}
                    >
                      <AmountInput
                        className={`w-[100px]`}
                        state={fixedValueState}
                        onAccept={handleFixedAmount}
                      />
                      <button
                        className={`flex items-center pl-4 pr-2 `}
                        onClick={() =>
                          handleFixedAmount(fGet(params.spendingFloor) + 5000)
                        }
                      >
                        <FontAwesomeIcon className="text-base" icon={faPlus} />
                      </button>
                      <button
                        className={`flex items-center px-2 `}
                        onClick={() =>
                          handleFixedAmount(
                            Math.max(0, fGet(params.spendingFloor) - 5000)
                          )
                        }
                      >
                        <FontAwesomeIcon className="text-base" icon={faMinus} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </RadioGroup.Option>
          </div>
        </RadioGroup>
      </ParamsInputBody>
    )
  }
)

const _roundUp = (x: number, step: number) => {
  const xPlusStep = x + step
  return xPlusStep - (xPlusStep % step)
}
